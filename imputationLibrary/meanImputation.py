import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def inputData(df_to_input):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df_to_input)
    df_complete = imp.transform(df_to_input)
    df_complete_final = pd.DataFrame(data=df_complete, index=df_to_input.index, columns=df_to_input.columns)
    return df_complete_final

def inputTestData(test_df, training_df, ignore_index=False): 
    for index, row in test_df.iterrows():
        training_df = training_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            training_df = inputData(training_df)
    return training_df[-test_df.shape[0]:]

def inputTrainingData(training_df):
    return inputData(training_df)

def input(training_df, test_df):
    df_training = inputTrainingData(training_df)
    df_test = inputTestData(test_df, training_df)
    return df_training, df_test
