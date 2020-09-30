import numpy as np
import random

def inputData(training_df):
    for index, row in training_df.iterrows():
        if (len(row)-row.count())>=1:
            nan_columns = np.argwhere(np.isnan(row))
            for col in nan_columns[0]:
                rand_row = index
                while(np.isnan(training_df.iloc[rand_row, col])):
                    rand_row = random.randint(0, training_df.shape[0]-1)
                training_df.iloc[index, col] = training_df.iloc[rand_row, col]
    return training_df

def inputTrainingData(training_df):
    return inputData(training_df)

def inputTestData(test_df, training_df, ignore_index=False):
    for index, row in test_df.iterrows():
        training_df = training_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            training_df = inputData(training_df)
    return training_df[-test_df.shape[0]:]

def input(df_training, df_test):
    return inputTrainingData(df_training), inputTestData(df_test, df_training)