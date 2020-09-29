import pandas as pd

def inputTrainingData(df):
    return df.ffill()

def inputTestData(test_df, training_df, ignore_index=False):
    for index, row in test_df.iterrows():
        training_df = training_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            training_df = training_df.ffill()
    return training_df[-test_df.shape[0]:]

def input(df_training, df_test):
    return inputTrainingData(df_training), inputTestData(df_test)