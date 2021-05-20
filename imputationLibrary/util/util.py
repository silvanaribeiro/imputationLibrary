import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import gauss
from datetime import datetime
import random

lower_bound = 1
upper_bound = 10
lower_bound2 = 100
upper_bound2 = 200
upper_bound_white_noise = 2
datelist = pd.date_range(datetime.today(), periods=1000).tolist()

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

def generate_white_noise(index):
    white_noise = [gauss(0.0, 1.0) for i in range(len(index))]
    white_noise = pd.DataFrame(white_noise, index=index)
    return white_noise.iloc[:,0]

def generate_seasonal(index):
    k_s = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    p = random.randint(lower_bound, upper_bound)
    #print(k_s, k_w, p)

    white_noise = generate_white_noise(index)
    x = np.arange(0, len(index)/10, 0.1)
    y = np.sin(p*x)*k_s
    seasonal = y + white_noise*k_w
    return seasonal

def generate_trended(index):
    k_t = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    #print(k_t, k_w)

    white_noise = generate_white_noise(index)
    x = np.arange(0, len(index)/10, 0.1)
    trend =k_t*x + k_w*white_noise
    return trend

def generate_trend_and_seasonal(index):
    k_s = random.randint(lower_bound2, upper_bound2)
    k_t = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    p = random.randint(lower_bound, upper_bound)
    #print(k_s, k_t, k_w, p)

    white_noise = generate_white_noise(index)
    x = np.arange(0, len(index)/10, 0.1)
    y = np.sin(p*x)*k_s
    seasonal_and_trend = y + k_w*white_noise + x*k_t
    return seasonal_and_trend