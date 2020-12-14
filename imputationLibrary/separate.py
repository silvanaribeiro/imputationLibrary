import pandas as pd
from imputationLibrary import nature
import random

def separate(df):
    df_white_noise = pd.DataFrame(index = df.index)
    df_seasonal = pd.DataFrame(index = df.index)
    df_trended = pd.DataFrame(index = df.index)
    df_trend_and_seasonal = pd.DataFrame(index = df.index)
    for col in df.columns:
        if nature.isWhiteNoise(df.loc[:,col]):
            df_white_noise = df_white_noise.join(df.loc[:,col])
        elif nature.isTrendedAndSeasonal(df.loc[:,col]):
            df_trend_and_seasonal = df_trend_and_seasonal.join(df.loc[:,col])
        elif nature.isTrended(df.loc[:,col]):
            df_trended = df_trended.join(df.loc[:,col])
        elif nature.isSeasonal(df.loc[:,col]):
            df_seasonal = df_seasonal.join(df.loc[:,col])
        else:
            print("Could not determine nature of the feature", col, "and it will be assigned to a group randomly.")
            rand = random.randint(1, 4)
            if rand == 1:
                df_white_noise = df_white_noise.join(df.loc[:,col])
            elif rand == 2:
                df_trend_and_seasonal = df_trend_and_seasonal.join(df.loc[:,col])
            elif rand == 3:
                df_trended = df_trended.join(df.loc[:,col])
            else:
                df_seasonal = df_seasonal.join(df.loc[:,col])
    return df_white_noise, df_seasonal, df_trended, df_trend_and_seasonal