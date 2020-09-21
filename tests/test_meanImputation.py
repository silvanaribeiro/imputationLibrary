import pandas as pd
import numpy as np
from imputationLibrary import meanImputation

def test_inputTrainingData():
    df_before = pd.DataFrame([np.nan,1,2,np.nan,0,0,0,0])
    df_after = pd.DataFrame([0.5,1,2,0.5,0,0,0,0])
    result, _ = meanImputation.inputTrainingData(df_before)
    assert df_after.equals(result)

def test_inputTestData():
    df_before = pd.DataFrame([np.nan,1,2,np.nan,0,0,0,0])
    result, imp = meanImputation.inputTrainingData(df_before)
    df_test = pd.DataFrame([np.nan, 1, 1, np.nan, 1, 1, 0, 0])
    df_after = pd.DataFrame([0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0])
    assert df_after.equals(meanImputation.inputTestData(df_test, imp))