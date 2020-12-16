from imputationLibrary import nature
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
max_iter = 50

def generate_white_noise():
    datelist = pd.date_range(datetime.today(), periods=1000).tolist()
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

def test_isWhiteNoise():
    result = {
        'error_white_noise':0,
        'error_seasonal':0,
        'error_trended':0,
        'error_trend_and_seasonal':0
        }

    for i in range(0,max_iter):

        white_noise = generate_white_noise()
        if not nature.isWhiteNoise(white_noise):
            result['error_white_noise']+=1

        seasonal = generate_seasonal()
        if nature.isWhiteNoise(seasonal):
            result['error_seasonal']+=1

        trend = generate_trended()
        if nature.isWhiteNoise(trend):
            result['error_trended']+=1

        trend = generate_trended2()
        if nature.isWhiteNoise(trend):
            result['error_trended']+=1

        trend_and_seasonal = generate_trend_and_seasonal()
        if nature.isWhiteNoise(trend_and_seasonal):
            result['error_trend_and_seasonal']+=1

    print(result)
    assert result['error_white_noise']<=0.10*max_iter
    assert result['error_trended']/2<=0.10*max_iter 
    assert result['error_seasonal']<=0.10*max_iter
    assert result['error_trend_and_seasonal']<=0.10*max_iter
    #assert True==False

def test_isSeasonal():
    result = {
        'error_white_noise':0,
        'error_seasonal':0,
        'error_trended':0,
        'error_trend_and_seasonal':0
        }
    for i in range(0,max_iter):

        white_noise = generate_white_noise()
        if nature.isSeasonal(white_noise):
            result['error_white_noise']+=1

        seasonal = generate_seasonal()
        if not nature.isSeasonal(seasonal):
            result['error_seasonal']+=1

        trend = generate_trended()
        if nature.isSeasonal(trend):
            result['error_trended']+=1

        #trend = generate_trended2()
        #assert False == nature.isSeasonal(trend)

        trend_and_seasonal = generate_trend_and_seasonal()
        if not nature.isSeasonal(trend_and_seasonal):
            result['error_trend_and_seasonal']+=1

    print(result)
    assert result['error_white_noise']<=0.10*max_iter
    assert result['error_trended']<=0.10*max_iter 
    assert result['error_seasonal']<=0.10*max_iter
    assert result['error_trend_and_seasonal']<=0.10*max_iter
    #assert True==False


def test_isTrended():
    result = {
        'error_white_noise':0,
        'error_seasonal':0,
        'error_trended':0,
        'error_trend_and_seasonal':0
        }

    for i in range(0,max_iter):

        white_noise = generate_white_noise()
        if nature.isTrended(white_noise):
            result['error_white_noise']+=1

        seasonal = generate_seasonal()
        if nature.isTrended(seasonal):
            result['error_seasonal']+=1
        #assert False == nature.isTrended(seasonal)

        trend = generate_trended()
        if not nature.isTrended(trend):
            result['error_trended']+=1

        trend = generate_trended2()
        if not nature.isTrended(trend):
            result['error_trended']+=1

        trend_and_seasonal = generate_trend_and_seasonal()
        if not nature.isTrended(trend_and_seasonal):
            result['error_trend_and_seasonal']+=1

    print(result)
    assert result['error_white_noise']<=0.10*max_iter
    assert result['error_trended']/2<=0.10*max_iter 
    assert result['error_seasonal']<=0.10*max_iter
    assert result['error_trend_and_seasonal']<=0.10*max_iter
    #assert True==False


def test_isTrendedAndSeasonal():
    result = {
        'error_white_noise':0,
        'error_seasonal':0,
        'error_trended':0,
        'error_trend_and_seasonal':0
        }

    for i in range(0,max_iter):

        white_noise = generate_white_noise()
        if nature.isTrendedAndSeasonal(white_noise):
            result['error_white_noise']+=1

        seasonal = generate_seasonal()
        if nature.isTrendedAndSeasonal(seasonal):
            result['error_seasonal']+=1
        #assert False == nature.isTrendedAndSeasonal(seasonal)

        trend = generate_trended()
        if nature.isTrendedAndSeasonal(trend):
            result['error_trended']+=1
        
        #trend = generate_trended2()
        #assert False == nature.isTrendedAndSeasonal(trend)

        trend_and_seasonal = generate_trend_and_seasonal()
        if not nature.isTrendedAndSeasonal(trend_and_seasonal):
            result['error_trend_and_seasonal']+=1

    print(result)
    assert result['error_white_noise']<=0.10*max_iter
    assert result['error_trended']<=0.10*max_iter 
    assert result['error_seasonal']<=0.10*max_iter
    assert result['error_trend_and_seasonal']<=0.10*max_iter
    #assert True==False