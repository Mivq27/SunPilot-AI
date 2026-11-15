import pandas as pd
import pytest
from src.recheck_metrics import evaluate_saved_predictions


def test_daytime_mae_uses_hour_end_labels_09_to_18():
    df=pd.DataFrame({
        'local_hour':range(8,19),
        'actual':[1000.0]+[1.]*10,
        'baseline':[0.0]*11,
        'ai_prediction':[0.0]+[1.]*10,
    })
    result=evaluate_saved_predictions(df)
    assert result['ai']['mae_08_18_w_m2']==0
    assert result['baseline']['mae_08_18_w_m2']==1
    assert result['ai']['mae_all_w_m2']==pytest.approx(round(1000/11,3))
