import pandas as pd

def read(file_name, parse_dates=True, index_col=0):
    df = pd.read_csv(file_name, parse_dates=parse_dates, index_col=index_col)
    return df