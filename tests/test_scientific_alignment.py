"""Scientific regression suite: prevents the v1.0 one-hour error returning."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.solar_time import historical_day, HOUR_END_LABELS
from src.scheduler import evaluate_frame, choose_best, solar_irradiation, bootstrap_relative_gain
from src.pv_scenario import hypothetical_self_use_kwh


def test_source_end_of_hour_labels_use_09_to_18_for_08_to_18_tasks():
    example = pd.DataFrame({
        'local_date': ['2025-07-01'] * 11,
        'local_hour': list(range(8, 19)),
        'actual': [1000] + list(range(9, 19)),
    })
    selected = historical_day(example, '2025-07-01')
    assert selected['local_hour'].tolist() == list(range(9, 19))
    assert selected['actual'].tolist() == list(range(9, 19))
    # Noon to 14:00 reads [12,13) and [13,14): 13:00,14:00 end-labels.
    assert solar_irradiation(selected['actual'].to_numpy(), 12) == 13 + 14


def test_historical_day_rejects_missing_hour_and_duplicates():
    d = pd.DataFrame({'local_date':['2025-01-01']*9,'local_hour':list(range(9,18))})
    with pytest.raises(ValueError,match='Incomplete'):
        historical_day(d,'2025-01-01')
    d2 = pd.DataFrame({'local_date':['2025-01-01']*11,'local_hour':list(range(9,19))+[10]})
    with pytest.raises(ValueError,match='duplicated'):
        historical_day(d2,'2025-01-01')


def test_first_high_value_on_old_08_label_must_not_change_decisions():
    d = pd.DataFrame({
        'local_date': ['2025-06-09']*11,
        'local_hour': list(range(8,19)),
        'actual': [1e6] + [0, 0, 0, 0, 0, 40, 55, 10, 0, 0],
        'baseline': [1e6] + [0, 0, 0, 0, 0, 40, 55, 10, 0, 0],
        'ai_prediction': [1e6] + [0, 0, 0, 0, 0, 40, 55, 10, 0, 0],
    })
    per_day, metrics = evaluate_frame(d, 2025)
    assert metrics['days'] == 1
    assert per_day.iloc[0]['ai_start'] == 13   # labels 14,15 = [13,15)
    assert per_day.iloc[0]['ai_actual_wh_m2'] == 95
    assert per_day.iloc[0]['oracle_actual_wh_m2'] == 95


def test_two_hour_fixed_noon_is_not_an_oracle():
    v = np.array([0, 0, 0, 30, 40, 200, 170, 10, 0, 0],dtype=float)
    assert solar_irradiation(v, 12) == 40+200  # slots [12,13),[13,14)
    assert choose_best(v) == 13
    assert solar_irradiation(v, choose_best(v)) == 370


def test_evaluate_frame_excludes_wrong_year():
    rows=[]
    for yr in ('2024','2025'):
        for hh in range(9,19):
            rows.append({'local_date':f'{yr}-07-01','local_hour':hh,
                         'actual':float(hh),'baseline':float(hh),'ai_prediction':float(hh)})
    _, metrics = evaluate_frame(pd.DataFrame(rows),2025)
    assert metrics['days']==1


def test_paired_bootstrap_reproducible_with_no_new_test_fit():
    ai=np.array([150.,180.,140.,200.])
    b=np.array([130.,170.,130.,190.])
    one=bootstrap_relative_gain(ai,b,seed=42,repetitions=100)
    two=bootstrap_relative_gain(ai,b,seed=42,repetitions=100)
    assert one==two
    assert len(one)==2 and 0<one[0]<one[1]


def test_hypothetical_pv_scenario_saturates_at_load():
    # 1000 W/m² with area 4m², 20% = 800W available, but 300W load.
    assert hypothetical_self_use_kwh([1000,1000]) == pytest.approx(0.6)
    assert hypothetical_self_use_kwh([0,0]) == 0
    with pytest.raises(ValueError):
        hypothetical_self_use_kwh([np.nan])


def test_local_daytime_avoids_dst_transition_hour():
    # Berlin DST transition is at 02:00, not in the daytime 08–18 region.
    dates = pd.date_range('2025-03-30T00:00:00Z', periods=24, freq='h')
    local = dates.tz_convert('Europe/Stockholm')
    d = pd.DataFrame({'local_date':local.strftime('%Y-%m-%d'),
                      'local_hour':local.hour,'actual':range(len(dates))})
    selected = historical_day(d,'2025-03-30')
    assert selected['local_hour'].tolist() == HOUR_END_LABELS
