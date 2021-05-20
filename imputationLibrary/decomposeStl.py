import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

def decomposeStl(ts):
    result = STL(ts).fit()
    return result