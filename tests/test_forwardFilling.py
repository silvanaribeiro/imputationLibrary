import pandas as pd
import numpy as np
from imputationLibrary import forwardFilling

def test_inputTrainingData():
    df_before = pd.DataFrame([1.0, 1, 2, np.nan, 0, 0, 0, 0])
    df_after = pd.DataFrame([1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0,0.0])
    print(forwardFilling.inputTrainingData(df_before))
    assert df_after.equals(forwardFilling.inputTrainingData(df_before))

def test_inputTestData():
    df_training = pd.DataFrame([1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0,0.0])
    df_test = pd.DataFrame([np.nan, 1, 1, np.nan, 1, 1, 0, 0])
    df_after = pd.DataFrame([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    result = forwardFilling.inputTestData(df_test, df_training, ignore_index=True)
    result = result.reset_index()
    assert df_after[:][0].equals(result[:][0])