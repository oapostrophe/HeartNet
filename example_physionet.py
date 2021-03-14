import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        """ Original code: gets entire 100hz dataset
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        """

        # New code: Get just one record instead
        data = [wfdb.rdsamp("./records100/00000/00001_lr")]
    else:
        """ Original code: gets entire 500hz dataset
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        """

        # New code: Get just one record instead
        data = [wfdb.rdsamp("./records500/00000/00001_hr")]

    data = np.array([signal for signal, meta in data])
    return data


# Original code: path = 'path/to/ptbxl/'
# New code: replaced with correct path
path = "./"
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

"""
Original code: splits data into train and test set
Commented out because it doesn't work when you only get one patient record

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
"""

# New code: Print shape of np array with data
print("Shape of data:")
print(np.shape(X))

# New code: Plot 400 samples (4 seconds) from lead 0, patient 0
import matplotlib.pyplot as plt
y_axis = X[0, :400, 0]
x_axis = np.arange(400)
plt.plot(x_axis, y_axis)