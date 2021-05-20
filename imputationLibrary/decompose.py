import statsmodels.api as sm

def additive(ts, period=None):
    if period:
        decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=period)
    else:
        decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    return decomposition

def multiplicative(ts):
    decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')
    return decomposition

def decompose(ts, period):
    return additive(ts, period)
    #try:
    #    decomposed = multiplicative(ts)
    #    print("Multiplicative Decomposition performed.")
    #    return decomposed
    #except:
    #    print("Additive Decomposition performed.")
    #    return additive(ts)
