import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def inputTrainingData(df_to_input):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df_to_input)
    df_complete = imp.transform(df_to_input)
    df_complete_final = pd.DataFrame(data=df_complete, index=df_to_input.index, columns=df_to_input.columns)
    return df_complete_final, imp

def inputTestData(test_df, imp): 
    df_test_imputed = imp.transform(test_df)
    df_test_final = pd.DataFrame(data=df_test_imputed, index=test_df.index, columns=test_df.columns)
    return df_test_final

def input(training_df, test_df):
    df_training, imp = inputTrainingData(training_df)
    df_test = inputTestData(test_df, imp)
    return df_training, df_test
