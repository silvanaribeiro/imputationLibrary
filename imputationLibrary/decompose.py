import statsmodels.api as sm

def additive(ts, period=None):

    if period:
        decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=period)
    else:
        decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    return decomposition

def multiplicative(ts, period=None):
    if period:
        decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative', period=period)
    else:
        decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')
    return decomposition

def decompose(ts, period=None):
    try:
        decomposed = multiplicative(ts, period)
        #print("Multiplicative Decomposition performed.")
        return decomposed, "multiplicative"
    except Exception as e:
        print(e)
        #print("Additive Decomposition performed.")
        return additive(ts, period), "additive"
