from imputationLibrary import separate
import numpy as np
import pandas as pd
from random import gauss
from datetime import datetime
import random

lower_bound = 1
upper_bound = 10
lower_bound2 = 100
upper_bound2 = 200
upper_bound_white_noise = 2
datelist = pd.date_range(datetime.today(), periods=1000).tolist()

def generate_white_noise():
    white_noise = [gauss(0.0, 1.0) for i in range(1000)]
    white_noise = pd.DataFrame(white_noise, index=datelist)
    return white_noise.iloc[:,0]

def generate_seasonal():
    k_s = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    p = random.randint(lower_bound, upper_bound)
    #print(k_s, k_w, p)
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(p*x)*k_s
    seasonal = y + white_noise*k_w
    return seasonal 

def generate_trended():
    k_t = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    #print(k_t, k_w)
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    trend =k_t*x + k_w*white_noise
    return trend

def generate_trended2():
    k_t = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    #print(k_t, k_w)
    white_noise = generate_white_noise()
    xlim = 50
    x = np.arange(0, xlim, 0.1)
    x2 = np.flip(x)
    x = np.append(x,x2)
    trend = k_t*x + k_w*white_noise
    return trend

def generate_trend_and_seasonal():
    k_s = random.randint(lower_bound2, upper_bound2)
    k_t = random.randint(lower_bound, upper_bound)
    k_w = random.randint(lower_bound, upper_bound_white_noise)
    p = random.randint(lower_bound, upper_bound)
    #print(k_s, k_t, k_w, p)
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(p*x)*k_s
    seasonal_and_trend = y + k_w*white_noise + x*k_t
    return seasonal_and_trend


def test_separate():
    # Não ta funcionando por causa do index. tem que dar um jeito de colocar o mesmo index pra todo mundo
    # talvez transformar pra pandas com o index do original??
    df = generate_white_noise()
    df = df.to_frame()
    df = df.join(generate_trend_and_seasonal(), lsuffix='_1', rsuffix='_2')
    df = df.join(generate_seasonal())
    df = df.join(generate_trended(), lsuffix='_3', rsuffix='_4')
    df = df.join(generate_trended())
    df = df.join(generate_white_noise(), lsuffix='_5', rsuffix='_6')
    df = df.join(generate_seasonal())
    df = df.join(generate_white_noise(), lsuffix='_7', rsuffix='_8')
    df = df.join(generate_white_noise())
    df = df.join(generate_trend_and_seasonal(), lsuffix='_9', rsuffix='_10')
    df = df.join(generate_seasonal())
    df = df.join(generate_trend_and_seasonal(), lsuffix='_11', rsuffix='_12')

    df_white_noise, df_seasonal, df_trended, df_trend_and_seasonal = separate.separate(df)

    assert len(df_white_noise.columns) == 4
    assert len(df_trended.columns) == 2
    assert len(df_seasonal.columns) == 3
    assert len(df_trend_and_seasonal.columns) == 3

