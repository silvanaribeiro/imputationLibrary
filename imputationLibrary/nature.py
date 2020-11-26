from imputationLibrary import decompose
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftshift
from sklearn import preprocessing
import numpy as np
import pandas as pd

def isWhiteNoise(ts):
    normalized = preprocessing.scale(np.array(ts.fillna(0)))
    corr = signal.correlate(normalized, normalized, mode='full')
    corr = corr/len(ts)
    upper_limit = 2/np.sqrt(len(ts))
    lower_limit = -2/np.sqrt(len(ts))
    result = (np.logical_or(corr>upper_limit, corr<lower_limit))
    count = len(result[result==True])
    print(lower_limit, upper_limit, count, (0.05*len(ts)*2), len(ts))
    return count <= (0.05*len(ts)*2)

def isSeasonal(ts):
    normalized = preprocessing.scale(np.array(ts.fillna(0)))
    normalized = pd.DataFrame(normalized, index = ts.index)
    N=normalized.shape[0]
    result = decompose.decompose(normalized)
    yf = fft(result.seasonal)
    yplot = fftshift(yf)
    result = 1.0/N * np.abs(yplot)
    peaks, info = find_peaks(result, height=0.002)
    print(peaks, info)
    peaks = peaks.tolist()
    for peak in peaks:
        if peak <= (N/2 + int(N*.002)) and peak >= N/2 - int(N*.002):
            peaks.remove(peak)
    if len(peaks) >=2 and not isWhiteNoise(ts):
        return True
    return False

def isTrended(ts):
    result = decompose.decompose(ts)
    if result.trend.mean() > 0.1:
        return True
    return False

def isTrendedAndSeasonal(ts):
    if not isWhiteNoise(ts) and isTrended(ts) and isSeasonal(ts):
        return True
    return False






