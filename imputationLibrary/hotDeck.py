import pandas as pd
import numpy as np
import random

def euclideanDistance(dot_missing, df, axis=1):
    return np.sqrt(np.sum(np.power(np.subtract(dot_missing, df),2), axis=axis))

def sortByDistance(dict_df):
    return {k: v for k, v in sorted(dict_df.items(), key=lambda x: x[1])}
    
def inputTrainingData(training_df, use_rand=True):
    #print(training_df)
    for index, row in training_df.iterrows():
        if (len(row)-row.count())>=1:
            row_reshaped=row.to_numpy().reshape(1, row.shape[0])
            dict_df = dict(zip(euclideanDistance(row_reshaped, training_df), list(training_df.index)))
            sorted_dict = sortByDistance(dict_df)
            if use_rand:
                rand_pos = random.randint(0, int((len(dict_df)-1)*0.05))
            else:
                rand_pos = 1
            nan_columns = np.argwhere(np.isnan(row_reshaped))
            for col in nan_columns[0][1:]:
                training_df.at[index, col] = training_df[col][sorted_dict[list(sorted_dict)[rand_pos-1]]]
    return training_df
            

