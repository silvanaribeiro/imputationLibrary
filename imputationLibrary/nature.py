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
    if not isWhiteNoise(ts):
        normalized = preprocessing.scale(np.array(ts.fillna(0)))
        normalized = pd.DataFrame(normalized, index = ts.index)
        result = decompose.decompose(ts)
        corr = signal.correlate(result.trend, result.trend, mode='full')
        corr = corr/len(ts)
        peaks, info = find_peaks(corr, height=2/np.sqrt(len(ts)))
        peaks = peaks.tolist()
        if len(peaks) < 2:
            return(False)
        first_value = peaks[1]-peaks[0]
        for i in range(0, len(peaks)-1):
            if peaks[i+1]-peaks[i] > first_value + 2 or peaks[i+1]-peaks[i] < first_value - 2:
                return(False)
        return(True)
    return(False)


def isTrended(ts):
    result = decompose.decompose(ts)
    if result.trend.mean() > 0.1:
        return True
    return False

def isTrendedAndSeasonal(ts):
    if not isWhiteNoise(ts) and isTrended(ts) and isSeasonal(ts):
        return True
    return False






