import pandas as pd
import numpy as np
from imputationLibrary import randomSampleImputation

def test_inputTrainingData():
    df_before = pd.DataFrame([[np.nan, 2.0,], [1.0, 1.0,], [1.0, 4.0,], [1.0, np.nan,], [1.0, 0.0,], [1.0, 1.0,], [1.0, np.nan,]])
    result = randomSampleImputation.inputTrainingData(df_before)
    assert result.isnull().values.any() == False

def test_inputTestData():
    df_training = pd.DataFrame([[1.0, 1.0,], [1.0, 1.0,],[1.0, 1.0,],[1.0, 1.0,],[1.0, 1.0,],[1.0, 1.0,],[1.0, 1.0,]])
    df_test = pd.DataFrame([[np.nan, 2.0,], [1.0, 1.0,], [1.0, 4.0,], [1.0, np.nan,], [1.0, 0.0,], [1.0, 1.0,], [1.0, np.nan,]])
    result = randomSampleImputation.inputTestData(df_test, df_training, ignore_index=True)
    assert result.isnull().values.any() == False