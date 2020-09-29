import pandas as pd
import numpy as np
from imputationLibrary import splineInterpolation
import matplotlib.pyplot as plt

def test_inputData():
    x = np.arange(0, 2*np.pi, np.pi/2)
    y = np.sin(x)
    x_new = np.arange(0, 2*np.pi, np.pi/4)
    y_new = splineInterpolation.inputData(x, y, x_new)
    y_new = np.array(y_new)
    result= [-7.97856902e-17,  8.75000000e-01,  1.00000000e+00, 6.25000000e-01,
              2.22044605e-16, -6.25000000e-01, -1.00000000e+00, -8.75000000e-01]
    result = np.array(result)
    assert np.isclose(y_new, result, rtol=1e-08, atol=1e-08).all()==True
    x = [0, 1, 2, 4, 5, 6]
    y = [1., 1., 1., 0., 0., 0.]
    x_new = [3]
    y_new = splineInterpolation.inputData(x, y, x_new)
    assert np.isclose(y_new, [0.5], rtol=1e-08, atol=1e-08).all() == True


def test_inputTrainingData():
    training_df = pd.DataFrame([1,1,1,np.nan,0,0,0], index=[0,1,2,3,4,5,6])
    training_df_imputed = splineInterpolation.inputTrainingData(training_df)
    result = pd.DataFrame([1,1,1,0.5,0,0,0], index=[0,1,2,3,4,5,6])
    assert np.isclose(result, training_df_imputed, rtol=1e-08, atol=1e-08).all() == True

    training_df = pd.DataFrame([[1,1,1,np.nan,0,0,0], [1,1,1,1,0,0,np.nan]])
    training_df=training_df.T
    training_df_imputed = splineInterpolation.inputTrainingData(training_df)
    result = pd.DataFrame([[1,1,1,0.5,0,0,0], [1,1,1,1,0,0,3.8666666667]])
    result=result.T
    assert np.isclose(result, training_df_imputed, rtol=1e-08, atol=1e-08).all() == True