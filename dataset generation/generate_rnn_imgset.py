# Generate separate lead images for RNN

import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import wfdb

def load_raw_data(annotations, sampling_rate, path):
    """
    Loads raw data from ptb_xl dataset into a numpy array.
    
    Parameters:
    annotations (dataframe): dataframe with list of EKG record filenames in dataset
    sampling_rate (int): 100 to read downsampled files, 500 for full resolution
    path (str): path to ptb_xl dataset files

    Returns:
    np array: list of EKG records
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in annotations.filename_lr]      
    else:
        data = [wfdb.rdsamp(path+f) for f in annotations.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic):
    """
    Generates MI / Normal label for a record from more detailed annotation data.

    Parameters:
    y_dic (dict): dictionary with diagnostic classes for a record

    Return:
    (bool) True if record is an MI, False if normal.
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return "MI" in tmp


path = "/raid/heartnet/data/" #TODO: Update with path to the ptbx folder before running

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Load raw signal data
sampling_rate=100
X = load_raw_data(Y, sampling_rate, path)

# Generate images
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
mi = Y.index[Y["diagnostic_superclass"]]
for number in mi:
    source = path+"imgset_rnn/normal/pt_"+str(number)
    target = path+"imgset_rnn/mi/"
    shutil.move(source, target)