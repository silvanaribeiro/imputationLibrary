from imputationLibrary import decompose
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftshift
from sklearn import preprocessing # maybe use StandardScaler?
import numpy as np
import pandas as pd
import math

def isWhiteNoise(ts):
    normalized = preprocessing.scale(np.array(ts.fillna(0)))
    corr = signal.correlate(normalized, normalized, mode='full')
    corr = corr/len(ts)
    upper_limit = 2/np.sqrt(len(ts))
    lower_limit = -2/np.sqrt(len(ts))
    result = (np.logical_or(corr>upper_limit, corr<lower_limit))
    count = len(result[result==True])
   #print(lower_limit, upper_limit, count, (0.05*len(ts)*2), len(ts))
    return count <= (0.05*len(ts)*2)

def isSeasonal(ts):
    count_outside_bounderies = 0
    if isWhiteNoise(ts):
        #print('é white noise')
        return False
    
    normalized = preprocessing.scale(np.array(ts.fillna(0)))
    corr = signal.correlate(normalized, normalized, mode='full')
    corr = corr/len(ts)
    corr = pd.DataFrame(corr)
    corr = corr.rolling(window=2).mean()
    corr = corr.dropna()
    corr = corr.to_numpy()
    peaks, info = find_peaks(corr[:,0], height=2/np.sqrt(len(ts)))
    peaks = peaks.tolist()
    comparison = 0
    if len(peaks) < 2:
        #print('menos de dois picos')
        return False
    elif len(peaks) >=6:
        comparison = peaks[5]-peaks[4]
    else:
        comparison = peaks[2]-peaks[1]

    for i in range(0, len(peaks)-1):
        if peaks[i+1]-peaks[i] > comparison + 2 or peaks[i+1]-peaks[i] < comparison - 2:
            count_outside_bounderies +=1


    if count_outside_bounderies > len(ts)*0.05:
        #print(ts)
        #print(peaks)
        #print(comparison)
        #print(count_outside_bounderies,len(ts)*0.05)
        #print('passou do numero de tolarancias')
        return False
    else:
        return True



def isTrended(ts):
    result = decompose.decompose(ts)
    if result.trend.mean() > 0.19:
        return True
    return False

def isTrendedAndSeasonal(ts):
    if not isWhiteNoise(ts) and isTrended(ts) and isSeasonal(ts):
        return True
    return False






