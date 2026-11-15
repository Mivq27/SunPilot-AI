"""The timestamp convention for historical hourly mean solar radiation.

Open-Meteo shortwave_radiation is a mean over the *preceding* hour.
E.g. an observation timestamped 09:00 refers to [08:00, 09:00).
A scheduled job [08:00, 10:00) must therefore use 09:00 and 10:00
hour-end labels, NOT 08:00 and 09:00.

This module is shared by the historical UI and the offline evaluation to
prevent silent one-hour differences between charts and backtests.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

LOCAL_START = 8
LOCAL_END = 18  # exclusive for actual execution time
HOUR_END_LABELS = list(range(LOCAL_START + 1, LOCAL_END + 1))


def historical_day(df: pd.DataFrame, date: str, *, date_col: str = 'local_date',
                   hour_col: str = 'local_hour') -> pd.DataFrame:
    """Select the ten 08:00–18:00 local hourly *intervals* from end labels.

    Input date/hour refer to the local wall clock of the **end timestamp**.
    This window excludes the DST transition hour (02:00), so these ten
    daytime values are unambiguous in Europe/Stockholm.
    """
    datetime.strptime(date, '%Y-%m-%d')
    required = {date_col, hour_col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing historical columns: {sorted(required - set(df.columns))}')
    selected = df.loc[(df[date_col] == date) &
                      df[hour_col].between(LOCAL_START + 1, LOCAL_END)].copy()
    selected = selected.sort_values(hour_col)
    hours = selected[hour_col].tolist()
    if hours != HOUR_END_LABELS:
        raise ValueError(
            f'Incomplete/duplicated solar intervals for {date}: expected end-hour labels '
            f'{HOUR_END_LABELS}, received {hours}'
        )
    return selected
