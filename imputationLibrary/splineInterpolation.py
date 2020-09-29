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
        if(len(x_new)>0):
            y_new = inputData(complete_df.index.values.tolist(), complete_df.iloc[:].values.tolist(), x_new.tolist())
            for i in range(0, len(x_new)):
                training_df.at[x_new[i], col] = y_new[i]
    return training_df

def inputTestData(test_df, training_df, ignore_index=False):
    for index, row in test_df.iterrows():
        training_df = training_df.append(row, ignore_index=ignore_index)
        if (len(row)-row.count())>=1:
            training_df = inputTrainingData(training_df)
    return training_df[-test_df.shape[0]:]