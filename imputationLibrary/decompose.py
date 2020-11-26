import statsmodels.api as sm

def additive(ts):
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    return decomposition

def multiplicative(ts):
    decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')
    return decomposition

def decompose(ts):
    try:
        return multiplicative(ts)
    except:
        return additive(ts)
