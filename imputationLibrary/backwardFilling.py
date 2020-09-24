import pandas as pd

def inputTrainingData(df):
    return df.bfill()

def inputTestData(df):
    return df.bfill()

def input(df_training, df_test):
    return inputTrainingData(df_training), inputTestData(df_test)