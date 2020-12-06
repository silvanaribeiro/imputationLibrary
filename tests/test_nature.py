from imputationLibrary import nature
import numpy as np
import pandas as pd
from random import gauss
from datetime import datetime
import random


def generate_white_noise():
    datelist = pd.date_range(datetime.today(), periods=1000).tolist()
    white_noise = [gauss(0.0, 1.0) for i in range(1000)]
    white_noise = pd.DataFrame(white_noise, index=datelist)
    return white_noise.iloc[:,0]

def generate_seasonal(k = 1, p = 1, k_w=1):
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(p*x)*k
    seasonal = y + white_noise*k_w
    return seasonal

def generate_trended(k=1, k_w=1):
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    trend =k*x + k_w*white_noise
    return trend

def generate_trended2(k=1, k_w=1):
    white_noise = generate_white_noise()
    xlim = 50
    x = np.arange(0, xlim, 0.1)
    x2 = np.flip(x)
    x = np.append(x,x2)
    trend = k*x + k_w*white_noise
    return trend

def generate_trend_and_seasonal(k_s=1, k_t=1, p=1, k_w=1):
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(p*x)*k_s
    seasonal_and_trend = y + k_w*white_noise + x*k_t
    return seasonal_and_trend

def test_isWhiteNoise():
    lower_bound = -100
    upper_bound = 100
    for i in range(0,10):

        k_s = random.randint(lower_bound, upper_bound)
        k_t = random.randint(lower_bound, upper_bound)
        k_w = random.randint(lower_bound, upper_bound)
        p = random.randint(lower_bound, upper_bound)
        
        white_noise = generate_white_noise()
        assert True == nature.isWhiteNoise(white_noise)

        seasonal = generate_seasonal(k_s, p, k_w)
        assert False == nature.isWhiteNoise(seasonal)

        trend = generate_trended(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend = generate_trended2(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend_and_seasonal = generate_trend_and_seasonal(k_s, k_t, p, k_w)
        assert False == nature.isWhiteNoise(trend_and_seasonal)

def test_isSeasonal():
    lower_bound = -100
    upper_bound = 100
    for i in range(0,10):

        k_s = random.randint(lower_bound, upper_bound)
        k_t = random.randint(lower_bound, upper_bound)
        k_w = random.randint(lower_bound, upper_bound)
        p = random.randint(lower_bound, upper_bound)
        
        white_noise = generate_white_noise()
        assert False == nature.isWhiteNoise(white_noise)

        seasonal = generate_seasonal(k_s, p, k_w)
        assert True == nature.isWhiteNoise(seasonal)

        trend = generate_trended(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend = generate_trended2(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend_and_seasonal = generate_trend_and_seasonal(k_s, k_t, p, k_w)
        assert True == nature.isWhiteNoise(trend_and_seasonal)


def test_isTrended():
    lower_bound = -100
    upper_bound = 100
    for i in range(0,10):

        k_s = random.randint(lower_bound, upper_bound)
        k_t = random.randint(lower_bound, upper_bound)
        k_w = random.randint(lower_bound, upper_bound)
        p = random.randint(lower_bound, upper_bound)
        
        white_noise = generate_white_noise()
        assert False == nature.isWhiteNoise(white_noise)

        seasonal = generate_seasonal(k_s, p, k_w)
        assert False == nature.isWhiteNoise(seasonal)

        trend = generate_trended(k_t, k_w)
        assert True == nature.isWhiteNoise(trend)

        trend = generate_trended2(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend_and_seasonal = generate_trend_and_seasonal(k_s, k_t, p, k_w)
        assert True == nature.isWhiteNoise(trend_and_seasonal)


def test_isTrendedAndSeasonal():
    lower_bound = -100
    upper_bound = 100
    for i in range(0,10):

        k_s = random.randint(lower_bound, upper_bound)
        k_t = random.randint(lower_bound, upper_bound)
        k_w = random.randint(lower_bound, upper_bound)
        p = random.randint(lower_bound, upper_bound)
        white_noise = generate_white_noise()
        assert False == nature.isWhiteNoise(white_noise)

        seasonal = generate_seasonal(k_s, p, k_w)
        assert False == nature.isWhiteNoise(seasonal)

        trend = generate_trended(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)
        
        trend = generate_trended2(k_t, k_w)
        assert False == nature.isWhiteNoise(trend)

        trend_and_seasonal = generate_trend_and_seasonal(k_s, k_t, p, k_w)
        assert True == nature.isWhiteNoise(trend_and_seasonal)