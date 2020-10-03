import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read(file_name, parse_dates=True, index_col=0):
    df = pd.read_csv(file_name, parse_dates=parse_dates, index_col=index_col)
    return df

def plot(real, missing, imputed):
    imputed_values_only = pd.DataFrame(columns=real.columns, index=real.index)
    imputed_indexes_only = missing.index[np.isnan(missing['0']) == True].tolist()
    imputed_values_only.iloc[imputed_indexes_only] = imputed.iloc[imputed_indexes_only]
    plt.figure(figsize=(18,8))
    plt.plot(real.index, real, label="Real Data", linewidth=8, color='blue')
    plt.plot(missing.index, imputed, label="Missing Data", color='red')
    plt.plot(imputed.index, imputed_values_only, 'x', label="Imputed Data", color='orange')
    plt.legend(['Real Data', 'Missing Data', 'Imputed Data'])
    plt.show()