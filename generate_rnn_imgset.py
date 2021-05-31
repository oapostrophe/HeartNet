# Generate separate lead images for RNN


import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]      
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]

        # Alternate code to get just one record instead
        # data = [wfdb.rdsamp("./records500/00000/00001_hr")]

    data = np.array([signal for signal, meta in data])
    return data


# Update with path to the ptbx folder
path = "/raid/heartnet/data/"
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Get labels as MI / non-MI
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return "MI" in tmp

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


# Generate just first 100 records for testing
# first_100 = Y[Y.index < 100]

# Generate images
import cv2
import matplotlib.pyplot as plt
import os
import shutil

patient_number = 1

for patient in X:

    # Make pt directory
    target = path+"imgset_rnn/normal/pt_"+str(patient_number)
    os.mkdir(target)

    # Concatenate 12 leads into an image
    row_images = []
    for row in range(3):

        # Put 4 leads in each row
        col_images = []
        for col in range(4):
            lead_number = row*4 + col
            data = patient[ :, lead_number]

            # Plot with MatPlotLib
            fig = plt.figure(frameon=False)
            plt.plot(data) 

            # Remove borders, ticks, etc.
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            # Make image, resize, and convert to grayscale
            filename = target + "/" + str(lead_number) + ".png"
            fig.savefig(filename)
            plt.close(fig)

    patient_number += 1

# Divide up MI vs non-MI images into directories
import shutil

mi = Y.index[Y["diagnostic_superclass"]]

for number in mi:
    source = path+"imgset_rnn/normal/pt_"+str(number)
    target = path+"imgset_rnn/mi/"
    shutil.move(source, target)