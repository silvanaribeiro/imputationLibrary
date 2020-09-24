import pandas as pd
import numpy as np
from imputationLibrary import hotDeck

def test_euclideanDistance():
    dot1 = pd.DataFrame([1,1,1,0,0,0,0])
    dot2 = pd.DataFrame([1,1,1,0,0,0,0])
    assert hotDeck.euclideanDistance(dot1, dot2, 0)[0] == 0.0 

    dot1 = pd.DataFrame([1,1,1,1,0,0,0])
    dot2 = pd.DataFrame([0,0,0,0,0,0,0])
    assert hotDeck.euclideanDistance(dot1, dot2, 0)[0] == 2

def test_sortByDistance():
    dict_before = {3:'a', 1:'b', 4:'c', 2:'d'}
    dict_after = {1:'b', 2:'d', 3:'a', 4:'c'}
    assert hotDeck.sortByDistance(dict_before) == dict_after

def test_inputTrainingData():
    df_before = pd.DataFrame([[1,1,1,np.nan,0,0,0], [1,1,1,1,0,0,0], [0,0,0,0,0,0,0]])
    df_after = pd.DataFrame([[1,1,1,1.0,0,0,0], [1,1,1,1.0,0,0,0], [0,0,0,0,0,0,0]])
    #print(df_before)
    result = hotDeck.inputTrainingData(df_before, use_rand = False)
    print(result)
    assert result.equals(df_after)