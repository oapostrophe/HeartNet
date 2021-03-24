import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        # Original code: gets entire 100hz dataset
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]

        """
        # New code: Get just one record instead
        data = [wfdb.rdsamp("./records100/00000/00001_lr")]
        """
    else:
        """ Original code: gets entire 500hz dataset
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        """

        # New code: Get just one record instead
        data = [wfdb.rdsamp("./records500/00000/00001_hr")]

    data = np.array([signal for signal, meta in data])
    return data


# Update with path to the ptbx folder
path = "../data/"
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

"""
Original code:
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))
"""
# Modified to store MI / non-MI:
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return "MI" in tmp

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

"""# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
"""

# Generate just first 100 records for testing
first_100 = Y[Y.index < 100]

# Generate images
import cv2
import matplotlib.pyplot as plt

patient_number = 1

for patient in X[]:

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
            filename = "../data/converted_imgs/" + str(lead_number) + ".png"
            fig.savefig(filename)
            plt.close(fig)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (512, 512), interpolation = cv2.INTER_LANCZOS4)

            # Add to images for current row
            col_images.append(im_gray)

        # Concatenate current row together, add to list of rows
        im_row = cv2.hconcat(col_images)
        row_images.append(im_row)

    # Concatenate all rows into final image and save
    im_final = cv2.vconcat(row_images)
    cv2.imwrite("../data/converted_imgs/pt_" + str(patient_number) + ".png", im_final)
    patient_number += 1

"""
# Outdated code for creating single image
# Create second image
y_axis = X[0, :, 2]
fig = plt.figure(frameon=False)
plt.plot(y_axis) 
plt.xticks([]), plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

filename = "./converted_imgs/2.png"
fig.savefig(filename)

im_gray2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
im_gray2 = cv2.resize(im_gray2, (512, 512), interpolation = cv2.INTER_LANCZOS4)
cv2.imwrite(filename, im_gray2)

# concatenate
im_v = cv2.hconcat([im_gray1, im_gray2]) 
# show the output image 
cv2.imwrite('./converted_imgs/final.png', im_v) 
"""

# Divide up MI vs non-MI images into directories
import shutil

mi = Y.index[Y["diagnostic_superclass"]]

for number in mi:
    source = "../data/normal/pt_"+str(number)+".png"
    target = "../data/mi/"
    shutil.move(source, target)
