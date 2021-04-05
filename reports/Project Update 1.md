# Project Update 1

1. Software: We plan to use PyTorch and possibly FastAI.

2. Dataset: PTB Diagnostic EKG Database https://archive.physionet.org/physiobank/database/ptbdb/

3. We plan to use a Convolutional Neural Network.  Our inputs will be images of EKG recordings generated from the above dataset.  Our output will be a binary label "MI" (Myocardial Infarction, or heart attack) or "Healthy"

# Project Update 2

#### 1. Accomplished: ####
- [x] We were able to generate a dataset of labeled ("MI" or "NORM") EKG images. Those were made by plotting individual leads in matplotlib and appending them together. The labels were extracted from a CSV document, containing the raw EKG data we downloaded online. 
- [x] We used OpenCV to transform the plots to 512x512 grayscale images and append them into a 3 x 4 image for each patient similar to how an EKG printout is organized. 
- [x] We first generated images for individual patients, but were eventually able to generate and label images for all patients in the dataset. We are currently working on training a neural network using the HPC. 

#### 2. Issues Encountered #### 
* We were not able to find any datasets that contained EKG images. Instead there were mostly datasets with raw data, which we had to find a way to transform into images. 
* We also had some issues extracting the labels for the dataset from the CSV document; specifically, we struggled with figuring out how to filter out the correct labels from a python list containing the diagnoses using pandas. 
* We are also having issues with training the network, as we've been unable so far to have fast ai read the data.

#### 3. We are all aiming for an A. ####
