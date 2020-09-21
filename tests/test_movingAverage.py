import pandas as pd
import numpy as np
from imputationLibrary import movingAverage

def test_inputTrainingData():
    df_before = pd.DataFrame([np.nan,1,0,np.nan,1,0,0,0])
    df_after = pd.DataFrame([0.5,1,0,0.5,1,0,0,0])
    assert df_after.equals(movingAverage.inputTrainingData(df_before, 2))

def test_inputTestData():
    df_train = pd.DataFrame([0.5, 1, 0, 0.5, 1, 0, 0, 0])
    df_test = pd.DataFrame([np.nan, 1, 1, np.nan, 1, 1, 0, 0])
    df_after = pd.DataFrame([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    df_after = df_after[:][0]
    assert df_after.equals(movingAverage.inputTestData(df_train, df_test, 2))