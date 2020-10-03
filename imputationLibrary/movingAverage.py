import pandas as pd
import numpy as np

def inputTrainingData(training_df, num_values):
    df_to_input = training_df.copy()
    for index, row in df_to_input.iterrows():
        if pd.isna(row[0]):
            count_prev = 0
            mean = 0
            index_prev = df_to_input.index.get_loc(index)
            while count_prev < num_values:
                index_prev -= 1
                if index_prev < 0:
                    break
                if not pd.isna(df_to_input.iloc[index_prev][0]):
                    count_prev += 1
                    mean += df_to_input.iloc[index_prev]
            count_next = 0
            index_next = df_to_input.index.get_loc(index)
            while count_next < num_values:
                index_next += 1
                if index_next >= len(df_to_input):
                    break
                if not pd.isna(df_to_input.iloc[index_next][0]):
                    count_next += 1
                    mean += df_to_input.iloc[index_next]
            mean = mean / (count_prev+count_next)
            df_to_input.at[index, df_to_input.columns[0]]  = mean
    return df_to_input

def inputTestData(training_df, test_df, num_values, ignore_index=False):
    df_to_input = training_df.copy()
    for index, row in test_df.iterrows():
        df_to_input = df_to_input.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            df_to_input = inputTrainingData(df_to_input, num_values)
    return df_to_input[-test_df.shape[0]:]

def input(training_df, test_df, num_values, ignore_index=False):
    training_complete = inputTrainingData(training_df, num_values)
    test_complete = inputTestData(training_complete, test_df, num_values, ignore_index)

    return training_complete, test_complete

    
