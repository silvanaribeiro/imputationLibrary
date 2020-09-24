import pandas as pd
import numpy as np
from imputationLibrary import backwardFilling

def test_inputTrainingData():
    df_before = pd.DataFrame([np.nan,1,2,np.nan,0,0,0,0])
    df_after = pd.DataFrame([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0,0.0])
    print(df_after)
    print(backwardFilling.inputTrainingData(df_before))
    assert df_after.equals(backwardFilling.inputTrainingData(df_before))

def test_inputTestData():
    df_test = pd.DataFrame([np.nan, 1, 1, np.nan, 1, 1, 0, 0])
    df_after = pd.DataFrame([1, 1.0, 1.0, 1, 1.0, 1.0, 0.0, 0.0])
    assert df_after.equals(backwardFilling.inputTestData(df_test))