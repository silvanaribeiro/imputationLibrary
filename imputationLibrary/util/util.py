import pandas as pd
import matplotlib.pyplot as plt

def read(file_name, parse_dates=True, index_col=0):
    df = pd.read_csv(file_name, parse_dates=parse_dates, index_col=index_col)
    return df

def plot(real, missing, imputed):
    #plt.plot(real.index, real, '--', xnew, f(xnew), '-', missing.index, missing, '-')
    plt.figure(figsize=(18,8))
    plt.plot(real.index, real, label="Real Data")
    plt.plot(missing.index, missing, '--', label="Missing Data")
    plt.plot(imputed.index, imputed, 'x', label="Imputed Data")
    plt.legend(['Real Data', 'Missing Data', 'Imputed Data'])
    plt.show()