from scipy import interpolate
import numpy as np

def inputData(x, y, x_new):
    tck = interpolate.splrep(x, y, s=0)
    y_new = interpolate.splev(x_new, tck, der=0)
    return y_new

def inputTrainingData(training_df):
    for col, data in training_df.iteritems():
        complete_df = training_df[col].dropna()
        x_new = np.setdiff1d(training_df.index, complete_df.index)
        y_new = inputData(complete_df.index.values.tolist(), complete_df.iloc[:].values.tolist(), x_new.tolist())
        for i in range(0, len(x_new)):
            training_df.at[x_new[i], col] = y_new[i]
    return training_df