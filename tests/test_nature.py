from imputationLibrary import nature
import numpy as np
import pandas as pd
from random import gauss
from datetime import datetime


def generate_white_noise():
    datelist = pd.date_range(datetime.today(), periods=1000).tolist()
    white_noise = [gauss(0.0, 1.0) for i in range(1000)]
    white_noise = pd.DataFrame(white_noise, index=datelist)
    return white_noise.iloc[:,0]

def generate_seasonal():
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(x)*10
    #y = np.sin(x)*13
    seasonal = y + white_noise
    return seasonal

def generate_trended():
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    trend = x + white_noise
    #trend = 30*x + white_noise
    return trend

def generate_trend_and_seasonal():
    white_noise = generate_white_noise()
    xlim = 100
    x = np.arange(0, xlim, 0.1)
    y = np.sin(x)*10
    seasonal_and_trend = y + white_noise*3 + x
    #y = np.sin(x)*13
    #seasonal_and_trend = 30*y + white_noise*7 + x
    return seasonal_and_trend

def test_isWhiteNoise():
    white_noise = generate_white_noise()
    assert True == nature.isWhiteNoise(white_noise)

    seasonal = generate_seasonal()
    assert False == nature.isWhiteNoise(seasonal)

    trend = generate_trended()
    assert False == nature.isWhiteNoise(trend)

    trend_and_seasonal = generate_trend_and_seasonal()
    assert False == nature.isWhiteNoise(trend_and_seasonal)

def test_isSeasonal():
    white_noise = generate_white_noise()
    assert False == nature.isSeasonal(white_noise)

    seasonal = generate_seasonal()
    assert True == nature.isSeasonal(seasonal)

    trend = generate_trended()
    assert False == nature.isSeasonal(trend)

    trend_and_seasonal = generate_trend_and_seasonal()
    assert True == nature.isSeasonal(trend_and_seasonal)


def test_isTrended():
    white_noise = generate_white_noise()
    assert False == nature.isTrended(white_noise)

    seasonal = generate_seasonal()
    assert False == nature.isTrended(seasonal)

    trend = generate_trended()
    assert True == nature.isTrended(trend)

    trend_and_seasonal = generate_trend_and_seasonal()
    assert True == nature.isTrended(trend_and_seasonal)

def test_isTrendedAndSeasonal():
    white_noise = generate_white_noise()
    assert False == nature.isTrendedAndSeasonal(white_noise)

    seasonal = generate_seasonal()
    assert False == nature.isTrendedAndSeasonal(seasonal)

    trend = generate_trended()
    assert False == nature.isTrendedAndSeasonal(trend)

    trend_and_seasonal = generate_trend_and_seasonal()
    assert True == nature.isTrendedAndSeasonal(trend_and_seasonal)