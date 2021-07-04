# Improved iteration plotting with wfdb database

import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import wfdb
from wfdb.io._signal import downround, upround

def calc_ecg_grids(minsig, maxsig, sig_units, fs, maxt, time_units):
    """
    Modified function from WDFB-Python: https://github.com/MIT-LCP/wfdb-python
    Calculate tick intervals for ECG grids.
    - 5mm 0.2s major grids, 0.04s minor grids.
    - 0.5mV major grids, 0.125 minor grids.
    10 mm is equal to 1mV in voltage.

    Parameters:
    minsig (float): The min value of the signal.
    maxsig (float): The max value of the signal.
    sig_units (list): The units used for plotting each signal.
    fs (float): The sampling frequency of the record.
    maxt (float): The max time of the signal.
    time_units (str): The x axis unit. Allowed options are: 'samples', 'seconds', 'minutes', and 'hours'.

    Returns:
    major_ticks_x (ndarray): The locations of the major ticks on the x-axis.
    minor_ticks_x (ndarray): The locations of the minor ticks on the x-axis.
    major_ticks_y (ndarray): The locations of the major ticks on the y-axis.
    minor_ticks_y (ndarray): The locations of the minor ticks on the y-axis.
    """
    # Get the grid interval of the x axis
    if time_units == 'samples':
        majorx = 0.2 * fs
        minorx = 0.04 * fs
    elif time_units == 'seconds':
        majorx = 0.2
        minorx = 0.04
    elif time_units == 'minutes':
        majorx = 0.2 / 60
        minorx = 0.04/60
    elif time_units == 'hours':
        majorx = 0.2 / 3600
        minorx = 0.04 / 3600

    # Get the grid interval of the y axis
    if sig_units.lower()=='uv':
        majory = 500
        minory = 125
    elif sig_units.lower()=='mv':
        majory = 0.5
        minory = 0.125
    elif sig_units.lower()=='v':
        majory = 0.0005
        minory = 0.000125
    else:
        raise ValueError('Signal units must be uV, mV, or V to plot ECG grids.')

    major_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, majorx)
    minor_ticks_x = np.arange(0, upround(maxt, majorx) + 0.0001, minorx)

    major_ticks_y = np.arange(downround(minsig, majory),
                              upround(maxsig, majory) + 0.0001, majory)
    minor_ticks_y = np.arange(downround(minsig, majory),
                              upround(maxsig, majory) + 0.0001, minory)

    return (major_ticks_x, minor_ticks_x, major_ticks_y, minor_ticks_y)


def load_raw_data(annotations, path):
    """
    Load ekg records from files into list.

    Parameters:
    annotations (dataframe) - annotations for each record

    Returns:

    """
    return [wfdb.rdrecord(path+f) for f in annotations.filename_lr]


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


path = "/raid/heartnet/data/" #TODO: Before running, update with path to your ptbx folder

# Get EKG labels
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Load raw signal data
X = load_raw_data(Y, path)

# Generate images
patient_number = 1
for record in X:
    fig = wfdb.plot_wfdb(record, title=" ", time_units="seconds", ecg_grids="all", return_fig=True)
    childs = fig.get_children()[1:]
    for child in childs:

        # Normalize y limits, remove ticks
        axes = child.axes
        axes.set_ylim([-1, 1])
        axes.tick_params(bottom=False, top=False, right=False, left=False)

        # Modified ECG grid code from WFDB-Python: https://github.com/MIT-LCP/wfdb-python
        auto_xlims = axes.get_xlim()
        auto_ylims= axes.get_ylim()

        (major_ticks_x, minor_ticks_x, major_ticks_y,
            minor_ticks_y) = calc_ecg_grids(auto_ylims[0], auto_ylims[1],
                                            "mv", sampling_rate, auto_xlims[1],
                                            "samples")

        min_x, max_x = np.min(minor_ticks_x), np.max(minor_ticks_x)
        min_y, max_y = np.min(minor_ticks_y), np.max(minor_ticks_y)

        for tick in minor_ticks_x:
            axes.plot([tick, tick], [min_y,  max_y], c='#ededed',
                            marker='|', zorder=1)
        for tick in major_ticks_x:
            axes.plot([tick, tick], [min_y, max_y], c='#bababa',
                            marker='|', zorder=2)
        for tick in minor_ticks_y:
            axes.plot([min_x, max_x], [tick, tick], c='#ededed',
                            marker='_', zorder=1)
        for tick in major_ticks_y:
            axes.plot([min_x, max_x], [tick, tick], c='#bababa',
                            marker='_', zorder=2)

        # Plotting the lines changes the graph. Set the limits back
        axes.set_xlim(auto_xlims)
        axes.set_ylim(auto_ylims)

        # Remove axes
        axes.set_axis_off()

    fig.savefig("../data/imgset2/initial/pt_"+str(patient_number)+".png")
    patient_number += 1
    plt.close(fig)

# Move MI and non-MI images into separate directories
mi = Y.index[Y["diagnostic_superclass"]]
for number in mi:
    source = "../data/imgset2/normal/pt_"+str(number)+".png"
    target = "../data/imgset2/mi/"
    shutil.move(source, target)