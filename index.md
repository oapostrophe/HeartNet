# Abstract
The automated diagnosis of Myocardial Infarction (MI) in 12-lead Electrocardiograms (EKGs) could extend the proven lifesaving benefits of pre-hospital MI detection to many Emergency Medical Services (EMS) systems currently without them.  While neural networks have been developed that accurately diagnose MI, such networks are difficult to deploy in clinical practice due to operating on digitized signal data rather than images of EKG waveforms, of which no large publicly available dataset exists.  We create such a dataset by plotting digitized EKG records from the Physionet PTB-XL dataset to simulate images of 12-lead EKG printouts.  We then train a FastAI CNN learner on the resulting dataset, comparing the accuracy of different network architectures and hyperparameters.  We also attempt to train an RNN by inputting individual images of each EKG lead separately.  Our results include the generation of a dataset containing 21,837 simulated 12-lead EKG images labeled as “MI” or “Normal”, along with a version augmented with simulated shadows, an alternative version plotted directly with MatPlotLib, and a final version containing each lead in a separate image for use in an RNN.  Our trained CNN classifies this dataset achieves an accuracy of 90%, sensitivity 72%, and specificity of 83%, while our RNN fails to train successfully.  Similar accuracy is achieved on both the augmented dataset with shadows and alternate directly plotted version.  These results show the ability of a CNN to detect MI in images of 12-lead EKGs, indicating the potential for developing a CNN-based application to automatically perform such detection in clinical practice.  While the sensitivity of our trained model is not yet high enough for clinical use, our generated dataset allows future researchers to develop techniques improving on our results.  It is our hope that such efforts will eventually lead to the development of an model trained on real EKG images that can be deployed in EMS systems to substantially reduce mortality for MI patients.


# Introduction

Heart disease is consistently the number one cause of death in America, and 805,000 heart attacks occur in the US annually. [\[1\]](#Citations)  Effective heart attack treatment relies on rapid diagnosis through 12-lead EKG in order to minimize the time to reperfusion therapy, and one of the largest improvements in the speed of diagnosis and treatment has been achieved through pre-hospital EKG analysis by EMS personnel.  For instance, one study found pre-arrival EMS identification of ST-Elevated Myocardial Infarction (STEMI) through 12-lead EKG lowered patients’ 30-day mortality rates from 15.3% to 7.3%, and 5-year mortality rates from 20.6% to 11.6%. [\[2\]](#Citations)

Yet despite the well-established lifesaving effects of performing pre-hospital 12-lead EKGs, many EMS systems continue to rely on Basic Life Support-level (BLS) providers untrained in EKG interpretation.  For instance, the Detroit EMS system, which handles approximately 100,000 calls annually is predominately staffed by BLS units, and Michigan has a statewide shortage of hundreds of ALS-level paramedics who would be trained to perform EKGs. [\[3\]](#Citations)  The result is that MIs go undetected before hospital arrival, resulting in slower diagnosis, slower treatment, and higher mortality.

In such EMS systems where staff trained in EKG interpretation are unavailable, automatic computer interpretation of EKG could potentially fill in the gap to provide the lifesaving benefits of pre-hospital diagnosis to  heart attack patients.  However, current MI detection algorithms in use on EKG monitors are unreliable.  For instance, one study found that a commonly used EMS monitor identified just 58% of STEMIs, missing the diagnosis in the remaining 42% of STEMI patients. [\[4\]](#Citations)

The past few years have seen researchers achieve breakthrough accuracy levels in automatic EKG interpretation using neural-network based approaches, in many cases rivaling or surpassing the accuracy of clinical providers. [\[6\], \[7\], \[8\]](#Citations)  However, such algorithms have yet to be widely applied to actual healthcare practice.  One barrier to such application has been the fact that all such algorithms to our knowledge have been developed to interpret digital EKG signals, which in the field are only available to software directly interfacing with an EKG monitor.  Clinical implementation of such algorithms would either need to be done on proprietary software run on EKG monitors themselves, or else on hardware with the ability to interface directly with EKG monitors to receive digital signals.  In either case, such implementation would be difficult both for researchers and clinical practitioners.

We attempt to resolve this difficulty by testing an alternate approach to neural-network based STEMi detection using images.  While digital signals are difficult to directly obtain in clinical practice, EKG monitors are equipped with printing capabilities that are routinely used to record readings.  If neural networks can accurately detect STEMis in images of 12-lead EKGs, then such printouts could be used as the basis of classification independently of the specific EKG device.  For instance, we envision an image-based approach being the basis of a mobile phone application that would allow providers to easily take a photo of an EKG printout and have it automatically classified.   Such an application could be developed by any researchers without needing access to proprietary software, and could be applied in clinical practice by healthcare agencies without needing to purchase specialized equipment.  Thus the feasibility of an image-based classification approach for EKGs could represent a large step in bridging the gap between research achievements and clinical practice.

A challenge to this approach, however, is the lack of a publicly available dataset of annotated 12-lead EKG printouts.  To test the feasibility of our approach with publicily available data, we create a dataset by plotting each lead of digital signal-based readings and combining the plots from each lead into a single image.  The result is images that approximate 12-lead EKG printouts.

Another substantial challenge is the fact that, depending on the part of the heart affected by MI, diagnostic features such as ST-elevation can appear in different leads and thus different parts of the image.  This fact requires our neural network to recognize these features regardless of their location.  Furthermore, even baseline EKGs can look dramatically different from patient to patient due to a wide varieties of arrhythmias and the routine presence of noise in readings caused by electrical fluctuations and patient movement during the procedure.  We attempt to overcome these challenges by training on a large and diverse dataset.

Our finished neural network achieves a high level of accuracy, suggesting the feasibility of automatically classifying EKGs through images of printouts.  While our trained network's sensitivity still requires improvement, techniques from prior MI classification studies can likely be employed to make such improvements in future work.  These results support the development of an application for computer STEMi recognition that can be easily integrated into clinical practice, and suggest that such an application can extend the lifesaving benefits of pre-hospital EKG interpretation to patients currently unable to receive it.

Ethical concerns for our project include potential biases in our dataset, which could influence our model’s performance and thus patient outcomes.  Testing an application based on our model in clinical practice will also raise ethical concerns due to not only concerns for patient safety, but the difficulty of obtaining high-quality informed consent from patients suffering from an acute MI.  Finally, the use of neural networks to make clinical treatment decisions raises broader concerns about the influence such practices may have on healthcare system design and practice.

# Related Works:

In this section, we will discuss relevant research pertaining to the application of neural networks to STEMi detection in EKGs. Past studies using convolutional neural networks (CNNs) have achieved high accuracy and sensitivity classifying STEMi's from various sets of leads. A study using 4-lead EKGs, sub-2d convolutional layers, lead asymmetric pooling layers to combine data from the multiple leads to achieve a 96% accuracy classifying Anterior STEMis. The algorithm's real-time performance was tested on a lightweight embedded system and found to be suitable for application in EKG wearable technologies. [\[7\]](#Citations)

A study more pertinent to our project's scope combines image-based deep learniing techniques to improve detection accuracy of an important marker for detecting myocardial ischemia in EKGs: a ST depression change. The CNN created yields an average AUC at 89.6% from an independent set, a mean sensitivity rate at 84.4% and a specificity at 84.9% at selected optimal cutoff thresholds. [\[9\]](#Citations)

Another study using a CNN and 12-lead EKGs, proposed a performance optimization technique through two data pre-processing methods: noise reduction (notch filter and high pass filter) and pulse segmentation (via QRS complex detection). The 
preprocessing techniques improved the sensitivity, specificty, and area under the curve (AUC) of the receiver operatnig characteristic (ROC), enhanncing STEMi detection performance on a 275 EKG record dataset with 179 STEMis and 96 normal. [\[8\]](#Citations)

Due to the complexity of classifiers like CNNs and other neural networks, key decision makers like physicians and experienced clinicians stigmatize the black-box nature of neural network-based diagnoses. A more recent study using a ML fusion model consistening of Logistic Regression (LR), Artificial Neural Network (ANN), and Gradient Boosting Machine (GBM) and modified approach using 554 temporal-spatial features of 12-lead EKGs from a sample size of 1244 patients was able to achieve a 52% gain over commercial software ad 37% gain over "experienced" clinicians. From the study, the researchers concluded that linear classifiers like LR are just as effect as ANN, which lends the use of linear classification favorability in clinical practice [\[10\]](#Citations)

# Methods

## Dataset Generation:
To generate our image datasets, we use the PTB-XL dataset\[[11\], \[12\]](#Citations), a collection of 21,837 labeled EKG records published by Physionet [\[13\]](#Citations).  5,486 (25%) of these EKGs are labelled as indicating Myocardial Infarction, while the remainder are either labelled Normal (9,528 / 44%) or as displaying a different abnormality not indicative of Myocardial Infarction (7,326 / 34%); for our purposes, we consider both of the latter to be "Normal".  Each file is a digital recording of electrical activity in 12 standard ECG leads over 10 seconds, as would typically be seen in the emergency setting when diagnosing Myocardial Infarction.  While the initial recordings were made at 500 Hz, the dataset also offers versions downsampled to 100 Hz, which we use throughout this study due to its closer resemblance to the frequencies measured by heart monitors in emergent clinical practice.

We create an initial dataset by using Physionet's [WFDB-Python](https://github.com/MIT-LCP/wfdb-python) library to read the numerical data from each record into a 12x1000 numpy array.  We then plot the 1000 numerical data points from each lead with Matploblib.  We transform the resulting plot into a 512x512 pixel grayscale image with OpenCV, then after repeating this process for all 12 leads we concatenate the resulting images.  The result is one 3 x 4 image displaying all 12 leads for each patient (see figure in discussion).  We use pandas to extract labels from a CSV file in the initial dataset and move each image to a parent directory indicating its label as "mi" or "normal"; the result is "Dataset 1".

Iterating on our first generation efforts, we then generate a second dataset by using WFDB-Python's plotting functions instead of directly passing numerical data to Matplotlib.  We enable an option in these functions to draw a background grid similar to those typically seen in EKG printouts.  We modify the source code of the WFDB plotting functions to allow the editing of the resulting figure with Matplotlib, then standardize the vertical limits of each plot's display and remove figure features such as tick marks, legends, and axes labels.  As before, each file is moved to a parent directory indicating its label, resulting in "Dataset 2" (see figure in discussion).

We then take our second dataset and attempt to augment our initial data to simulate an irregularity commonly seen in mobile phone photographs of EKG printouts: shadow overlying the image.  We use the [Automold Road Image Augmentation Library](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library) to randomly add between 1 and 3 5-vertex shadows to each image in Dataset 2.  We save the results as new images, resulting in "Dataset 3" (see figure in discussion).

Finally, to test the classification of our images using an RNN, we generate a fourth dataset with each lead in its own image.  We use the same methods as for Datset 1, except stopping after generating a figure of each lead with Matplotlib.  No transformation is applied with OpenCV, nor are the images concatenated together.  The result is 12 images for each patient plotting each of the 12 EKG leads separately, making up "Dataset 4" (see figure in discussion).
To come up with our final set of hyperparameters to run on the entire imgset2, we first systematically tested the following resnets: resnet18, resnet34, resnet50. We controlled for the data set size (777 images in both mi and normal folders), batch size (16), epochs (20), and resizing (none). From the results, resnet34produced an error rate of 0.38 at epoch 12. Resnet34 also produced lower training losses and lower validation losses than the other resnets, so we chose resnet34 to systematically test batch size and resizing. In order to save time, we reduced the data set size from 777 to 400 images in each folder.

## Batch Size Testing:
To figure out which batch size produce the best results, we trained the model using 4, 8, 16, 24, 32, 40 and 64 for the batch size and controlled for the other hyperparameters (400 images in the dataset, 15 epochs, and no resizing). From a comparison of the metrics, we determined that 16, 32, and 64 produced the best results. Testing for larger batch sizes proved difficult as we ran into memory errors. However, through controlling the other hyperparameters and conducting resizing testing, we found that resizing had no impact on the results and thus no information is loss in the process of resizing. Thus we were able to resize the images down and increase the batch size to 128. Thus we were able to train the model using 4, 8, 16, 24, 32, 40, 64, and 128 for the batch sizes.

## Resizing Testing:
To figure out the impact of resizing on the results, we trained the model multiple times and changed the resize factors and controlling for the other hyperparameters. Sticking to a resnet34, batch size 16, epochs 15, we vary the resize factor to 300x300, 400x400, 500x500 and compare it to no resizing. Through the results we find that resizing has no impact on the results.

## Training on the Entire Dataset:
We train the model with the entire imgset2 using following sets of hyperparameters: The first trial we use a batch size of 16, 15 epochs, and no resizing (as resizing has no impact). The second trial we use a batch size of 32, 15 epochs, and no resizing. The third trial we use batch size of 64, 15 epochs, and no resizing.

To contextualize the performance of our CNN, we felt compelled to build and train another type of neural network. Ultimately, we decided on the RNN, which is distinguished by its memory capacity — when generating outputs, it recalls things learned from prior inputs. We believed that such a quality would be relevant to EKG classification, as Myocardial Infarction EKGs often contain complementary ST manifestations on the different leads (i.e., an ST elevation in one lead is coupled with an ST depression in other), and we hypothesized that a NN model with the ability to recall characteristics of previous leads would have an upper hand in MI diagnosis. 

The RNN training dataset was adapted from Dataset 4: we converted each patient's 12 lead images (i.e., the images in each subdirectory of Dataset 4) to tensors and stacked them, producing one tensor for every patient. Then, we saved each resulting tensor in a file and added a corresponding mi or normal label. 

To create the RNN, we first adapted a Pytorch implementation of a Convolutional LSTM ([https://github.com/ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)), specifying one LSTM layer, 10 hidden layers, and a kernel size of 3. Next, we used Pytorch to flatten the LSTM layer to a single tensor, before applying a linear transformation and a sigmoid activation function. For the RNN training process, we selected Binary Cross Entropy for our cost function, Adam for our optimization algorithm, two for our batch size (to accommodate our GPU memory constraints), and ten for the number of epochs.

# Discussion

## Datasets

Our first generated image dataset successfully plotted EKG waveforms and allowed training of a CNN; however, the generated images bore several striking differences from real-world EKG printouts (see figures below).  First, many images in our dataset show repeated large positive or negative vertical spikes not seen in normal EKG printouts.  While EKG waveforms do contain various peaks and valleys, these also exist in our dataset and bear a distinct shape from the spikes, which instead seem to be the result of single very high or low outlier data points.  EKG readings always contain some amount of noise or artifact, which may be the source of such outlier data points.  Other studies classifying EKG data have pre-processed data with notch or high-pass filters, which would remove extreme outliers; it's possible that clinical EKG monitoring equipment employs similar techniques that prevent such spikes from displaying on printouts.

The images in dataset 1 are also significantly more choppy than typically seen in real EKG printouts.  This can partially be explained by the sampling resolution of our data, which at 100 Hz is significantly higher than resolutions in the 40-60Hz range recorded by many clinical monitors; the result is less "smooth" lines as more fine-grain details of the waveform are captured.  It's also possible that other pre-processing techniques to smooth data are used by clinical equipment, in order to reduce the effects of noise and make waveforms easier for humans to read.  Other notable differences include the lack of a background grid in our images and the separation of each lead by a significant amount of whitespace.

Despite these differences, however, the fact that a CNN was able to classify these images with 89% accuracy indicates that these images successfully visualize the important diagnostic features in each record.  As such, the dataset successfully serves its purpose as a proof of concept for image-based classification of EKGs, despite its lack of realistic representation of the sort of images we were hoping to simulate.

Fig. 1: Image from Dataset 1
![Image from Dataset 1](images/dataset_1.png){:class="img-responsive"}

Fig. 2: Image of real EKG printout for comparison
![Real EKG for comparison](/images/real_ekg1.jpg)

Our second dataset addresses many of the issues with the first by utilizing the WFDB library's plotting functions, which eliminates both the large vertical spikes and much of the choppiness in the original images.  Not only does this dataset produce smoother and more realistic looking waveforms, but it also adds a background grid and removes most of the excess whitespace separating images.  However, dataset 2 did introduce a separate issue in the layout of each lead in the image.  While real EKG images are typically arranged in a 3 x 4 grid, these images vertically stack all leads in a single column.  We opted not to horizontally concatenate the leads from each image in this way due to finding that each plot was substantially wider than typically seen in EKG printouts, possibly indicating the 10-second samples in our data represent a longer timespan than is usually captured in a single printout.  While we considered taking a narrower subset of the image, doing so could possibly result in removing the section of the EKG where the key diagnostic features determining it as an MI are present.  Without beaty-by-beat annotation of our dataset, we opted to leave the images in a one-column format.  Again, this dataset was able to be classified with a high degree of accuracy by a CNN, indicating that it successfully visualized key diagnostic features and serves as proof of concept for image-based EKG classification.

Fig. 3: Image from dataset 2
![Image from Dataset 2](/images/dataset_2.png)

Our augmentation in the third dataset successfully simulated one of the most common distortions present in real pictures of EKG printouts: shadow overlaying the image.  While the shadows produced aren’t entirely realistic, they do provide a good test for the sort of artifact a classifier would have to overcome on EKG images captured during clinical practice.

Fig. 4: Image from dataset 3
![Image from Dataset 3](images/dataset_3.png)

Finally, our fourth dataset generated individual lead images similar to those that might be captured by photographing part of an EKG printout at a time.  These allowed us to feed sequential images into an RNN to test the performance of such a network on MI classification.

Fig. 5: Image from dataset 4
![Image from Dataset 4](images/dataset_4.png)

## CNN Training

### Resnet Comparisons:
Controls:
- no resizing
- batch size=16
- 20 epochs
- dataset size = 777

Resnet18
0         1.266850    2.942806    0.509677    0.000000      0.000000         0.000000  00:11     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.122408    3.563650    0.509677    0.000000      0.000000         0.000000  00:14     
1         0.972592    2.984641    0.509677    0.000000      0.000000         0.000000  00:14     
2         0.914277    1.436547    0.503226    0.012658      1.000000         0.025000  00:14     
3         0.883117    0.792957    0.538710    0.449367      0.470199         0.459547  00:14     
4         0.801197    1.671615    0.509677    0.000000      0.000000         0.000000  00:14     
5         0.751294    0.739595    0.480645    0.727848      0.520362         0.606860  00:14     
6         0.691240    0.879652    0.477419    0.955696      0.517123         0.671111  00:14     
7         0.647559    0.798806    0.470968    0.917722      0.521583         0.665138  00:14     
8         0.608896    0.942204    0.483871    1.000000      0.512987         0.678112  00:14     
9         0.594619    0.752372    0.474194    0.221519      0.593220         0.322581  00:14     
10        0.572300    0.749183    0.509677    0.000000      0.000000         0.000000  00:14     
11        0.543595    0.712606    0.490323    0.537975      0.518293         0.527950  00:14     
12        0.542646    0.692320    0.477419    0.582278      0.528736         0.554217  00:14     
13        0.513745    0.796773    0.500000    0.018987      1.000000         0.037267  00:22     
14        0.504420    0.751662    0.470968    0.917722      0.521583         0.665138  00:26     
15        0.485044    0.721429    0.445161    0.879747      0.538760         0.668269  00:28     
16        0.464817    0.718124    0.458065    0.841772      0.532000         0.651961  00:34     
17        0.469875    0.689644    0.445161    0.702532      0.549505         0.616667  00:39     
18        0.443754    0.679035    0.435484    0.715190      0.556650         0.626039  00:42     
19        0.432085    0.696283    0.441935    0.803797      0.545064         0.649616  00:39 

Resnet34
0         1.186446    2.405405    0.509677    0.000000      0.000000         0.000000  00:32                            
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.060453    0.879981    0.538710    0.379747      0.465116         0.418118  00:43                             
1         1.062914    1.046898    0.541936    0.822785      0.481481         0.607477  01:04                             
2         0.902046    0.667908    0.400000    0.575949      0.614865         0.594771  01:07                             
3         0.860821    1.142523    0.490323    1.000000      0.509677         0.675214  00:51                             
4         0.762702    0.846567    0.509677    0.000000      0.000000         0.000000  00:45                             
5         0.715962    0.883984    0.509677    0.000000      0.000000         0.000000  00:45                             
6         0.652512    0.762305    0.509677    0.000000      0.000000         0.000000  00:45                             
7         0.609759    0.900040    0.490323    1.000000      0.509677         0.675214  00:45                             
8         0.591689    0.737394    0.490323    1.000000      0.509677         0.675214  00:45                             
9         0.574536    0.829512    0.509677    0.000000      0.000000         0.000000  00:45                              
10        0.542365    0.707229    0.490323    1.000000      0.509677         0.675214  00:45                              
11        0.515140    0.879219    0.490323    1.000000      0.509677         0.675214  00:45                              
12        0.506055    0.675017    0.380645    0.474684      0.681818         0.559701  00:46                              
13        0.480450    0.687044    0.490323    1.000000      0.509677         0.675214  00:46                              
14        0.472038    0.725222    0.490323    1.000000      0.509677         0.675214  00:46                              
15        0.443922    0.679107    0.467742    0.943038      0.522807         0.672686  00:46                              
16        0.435891    0.696529    0.490323    1.000000      0.509677         0.675214  00:46                              
17        0.414416    0.685748    0.493548    0.993671      0.508091         0.672377  01:00                              
18        0.382245    0.730911    0.490323    1.000000      0.509677         0.675214  00:48                              
19        0.367419    0.727643    0.490323    1.000000      0.509677         0.675214  00:48   

Resnet50
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.326219    0.877586    0.490323    0.392405      0.525424         0.449275  00:27     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.020131    2.424108    0.490323    1.000000      0.509677         0.675214  00:35     
1         0.908129    2.651418    0.490323    1.000000      0.509677         0.675214  00:35     
2         0.867466    1.309385    0.490323    1.000000      0.509677         0.675214  00:35     
3         0.856429    1.031030    0.490323    1.000000      0.509677         0.675214  00:35   
5         0.722112    0.830483    0.509677    0.000000      0.000000         0.000000  00:35     
6         0.677697    0.698335    0.490323    1.000000      0.509677         0.675214  00:35     
7         0.636401    0.732785    0.490323    1.000000      0.509677         0.675214  00:35     
8         0.614662    0.746337    0.509677    0.000000      0.000000         0.000000  00:35     
9         0.595710    0.694422    0.490323    1.000000      0.509677         0.675214  00:35     
10        0.568519    0.685993    0.490323    1.000000      0.509677         0.675214  00:35     
11        0.533861    0.832971    0.490323    1.000000      0.509677         0.675214  00:35     
12        0.519946    0.711103    0.509677    0.000000      0.000000         0.000000  00:35     
13        0.480545    0.684510    0.496774    0.987342      0.506494         0.669528  00:35     
14        0.470463    0.704919    0.490323    1.000000      0.509677         0.675214  00:35     
15        0.443404    0.782374    0.490323    1.000000      0.509677         0.675214  00:35     
16        0.431273    0.895937    0.490323    1.000000      0.509677         0.675214  00:35     
17        0.441779    0.842597    0.490323    1.000000      0.509677         0.675214  00:35     
18        0.402331    0.890843    0.490323    1.000000      0.509677         0.675214  00:35     
19        0.379989    0.889967    0.490323    1.000000      0.509677         0.675214  00:35 

As we can see from the results, resnet34 generates the lowest error rate among the other resnets with an error rate of 0.38 at epoch 12.

#### Batch Size Testing:
Controls: 
- Resnet34
- no resizing
- 15 epochs
- dataset size = 400

Batch size = 16
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.267288    1.716300    0.353191    0.011905      1.000000         0.023529  00:09     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.041441    0.900239    0.485106    0.214286      0.272727         0.240000  00:11     
1         1.012747    1.682268    0.357447    0.000000      0.000000         0.000000  00:11     
2         0.932582    1.363935    0.357447    0.000000      0.000000         0.000000  00:11     
3         0.931104    2.164849    0.357447    0.000000      0.000000         0.000000  00:11     
4         0.788046    0.725249    0.361702    0.011905      0.333333         0.022989  00:11     
5         0.727885    0.746054    0.574468    0.809524      0.363636         0.501845  00:11     
6         0.683206    0.855712    0.357447    0.000000      0.000000         0.000000  00:11     
7         0.654375    0.718607    0.514894    0.761905      0.387879         0.514056  00:11     
8         0.631245    0.654714    0.348936    0.023810      1.000000         0.046512  00:11     
9         0.587375    1.038760    0.629787    0.988095      0.360870         0.528662  00:11     
10        0.547802    0.661423    0.365957    0.190476      0.470588         0.271186  00:11     
11        0.536663    0.777119    0.357447    0.000000      0.000000         0.000000  00:11     
12        0.536389    0.916005    0.357447    0.000000      0.000000         0.000000  00:11     
13        0.530064    0.879097    0.357447    0.000000      0.000000         0.000000  00:11     
14        0.521089    0.795361    0.357447    0.000000      0.000000         0.000000  00:11


training loss consistently decreases, validation loss hovers around 0.7
Batch size = 8
0         1.294832    0.846603    0.357447    0.000000      0.000000         0.000000  00:14     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.924234    0.731750    0.429787    0.130952      0.282051         0.178862  00:19     
1         0.996812    1.503362    0.646809    0.988095      0.354701         0.522013  00:19     
2         0.970935    1.544715    0.642553    1.000000      0.357447         0.526646  00:26     
3         0.872443    1.717994    0.642553    1.000000      0.357447         0.526646  00:35     
4         0.698454    0.941579    0.642553    1.000000      0.357447         0.526646  00:54     
5         0.675875    0.733263    0.357447    0.000000      0.000000         0.000000  00:55     
6         0.661197    0.668581    0.365957    0.154762      0.464286         0.232143  00:56     
7         0.625095    0.650854    0.357447    0.000000      0.000000         0.000000  00:55     
8         0.598482    0.846583    0.642553    0.988095      0.356223         0.523659  00:54     
9         0.551891    0.850611    0.642553    1.000000      0.357447         0.526646  00:54     
10        0.565568    1.314064    0.642553    1.000000      0.357447         0.526646  00:54     
11        0.549785    1.710631    0.642553    1.000000      0.357447         0.526646  00:55     
12        0.495891    1.286225    0.642553    1.000000      0.357447         0.526646  00:55     
13        0.481747    1.452635    0.642553    1.000000      0.357447         0.526646  00:55     
14        0.466517    1.391240    0.642553    1.000000      0.357447         0.526646  00:54  

Batch size = 4
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.252684    3.209037    0.642553    1.000000      0.357447         0.526646  00:32                              
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.989397    0.710874    0.429787    0.309524      0.376812         0.339869  00:58                               
1         0.914720    0.727190    0.514894    0.297619      0.287356         0.292398  01:06                               
2         0.858095    0.850841    0.591489    0.916667      0.368421         0.525597  01:07                               
3         0.809154    0.647854    0.357447    0.000000      0.000000         0.000000  01:06                             
4         0.793582    1.518062    0.642553    1.000000      0.357447         0.526646  01:07                               
5         0.742393    0.648858    0.353191    0.107143      0.529412         0.178218  01:06                               
6         0.694172    0.660459    0.327660    0.178571      0.652174         0.280374  01:06                               
7         0.708466    0.906908    0.642553    1.000000      0.357447         0.526646  01:07                               
8         0.638981    0.661673    0.336170    0.142857      0.631579         0.233010  01:07                               
9         0.642837    0.731711    0.646809    0.940476      0.349558         0.509677  00:54                                
10        0.611053    0.630330    0.348936    0.035714      0.750000         0.068182  00:45                                
11        0.630409    0.816694    0.638298    1.000000      0.358974         0.528302  00:41                                
12        0.556286    0.795529    0.642553    1.000000      0.357447         0.526646  00:24                                
13        0.581169    0.851194    0.642553    1.000000      0.357447         0.526646  00:25                                
14        0.518725    0.797145    0.642553    1.000000      0.357447         0.526646  00:25 

Batch size = 24
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.227526    1.306244    0.629787    0.964286      0.358407         0.522581  00:36     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.063386    3.524030    0.642553    1.000000      0.357447         0.526646  00:48     
1         0.998468    2.105664    0.642553    1.000000      0.357447         0.526646  00:48     
2         0.980653    0.684795    0.361702    0.166667      0.482759         0.247788  00:48     
3         0.909343    1.845695    0.642553    1.000000      0.357447         0.526646  00:49     
4         0.884520    3.141172    0.642553    1.000000      0.357447         0.526646  00:48     
5         0.786155    3.055175    0.642553    1.000000      0.357447         0.526646  00:49     
6         0.743422    1.972553    0.642553    1.000000      0.357447         0.526646  00:49     
7         0.689451    1.374256    0.642553    1.000000      0.357447         0.526646  00:49     
8         0.630955    1.747043    0.642553    1.000000      0.357447         0.526646  00:48     
9         0.595913    0.919722    0.642553    1.000000      0.357447         0.526646  00:48     
10        0.559691    0.993671    0.642553    1.000000      0.357447         0.526646  00:49     
11        0.535118    0.736718    0.612766    0.964286      0.364865         0.529412  00:43     
12        0.525502    0.866666    0.642553    1.000000      0.357447         0.526646  00:31     
13        0.508957    0.843448    0.642553    1.000000      0.357447         0.526646  00:32     
14        0.480275    0.851002    0.642553    1.000000      0.357447         0.526646  00:33 

Batch size = 32
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.194540    1.132789    0.348936    0.059524      0.625000         0.108696  00:12     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.189424    0.694222    0.434043    0.309524      0.371429         0.337662  00:17     
1         1.027123    0.806549    0.604255    0.904762      0.361905         0.517007  00:17     
2         0.993330    1.022487    0.642553    1.000000      0.357447         0.526646  00:17     
3         0.920281    0.979950    0.357447    0.000000      0.000000         0.000000  00:17     
4         0.877697    0.691950    0.357447    0.000000      0.000000         0.000000  00:17     
5         0.869074    0.788514    0.357447    0.000000      0.000000         0.000000  00:17     
6         0.798583    0.976212    0.357447    0.000000      0.000000         0.000000  00:17     
7         0.751028    0.868930    0.357447    0.000000      0.000000         0.000000  00:17     
8         0.694284    1.076957    0.357447    0.000000      0.000000         0.000000  00:17     
9         0.655459    1.018884    0.357447    0.000000      0.000000         0.000000  00:17     
10        0.609906    0.654968    0.353191    0.011905      1.000000         0.023529  00:17     
11        0.575430    0.643938    0.357447    0.000000      0.000000         0.000000  00:17     
12        0.560495    0.761334    0.357447    0.000000      0.000000         0.000000  00:17     
13        0.558590    0.741279    0.357447    0.000000      0.000000         0.000000  00:17     
14        0.539490    0.747357    0.357447    0.000000      0.000000         0.000000  00:17  

Batch size = 40
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.172498    1.300174    0.604255    0.916667      0.363208         0.520270  00:13     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.996883    0.680950    0.421277    0.404762      0.409639         0.407186  00:17     
1         0.975290    0.792021    0.523404    0.833333      0.391061         0.532319  00:17     
2         0.979466    0.798902    0.365957    0.011905      0.250000         0.022727  00:17     
3         0.959108    1.336064    0.634043    0.976190      0.358079         0.523962  00:17     
4         0.910689    2.428536    0.642553    1.000000      0.357447         0.526646  00:18     
5         0.847436    1.453672    0.642553    1.000000      0.357447         0.526646  00:17     
6        0.806403    1.253855    0.642553    1.000000      0.357447         0.526646  00:17     
7         0.765040    1.818839    0.642553    1.000000      0.357447         0.526646  00:17     
8         0.716208    0.699736    0.421277    0.297619      0.384615         0.335570  00:17     
9         0.701339    0.648663    0.361702    0.130952      0.478261         0.205607  00:17     
10        0.648997    0.651490    0.370213    0.142857      0.444444         0.216216  00:17     
11        0.605314    0.691765    0.438298    0.416667      0.393258         0.404624  00:17     
12        0.586192    0.669824    0.417021    0.321429      0.397059         0.355263  00:17     
13        0.558702    0.686057    0.446809    0.392857      0.379310         0.385965  00:17     
14        0.535528    0.658568    0.395745    0.190476      0.390244         0.256000  00:17

Batch size = 64
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.243336    0.949512    0.455319    0.821429      0.428571         0.563265  00:13     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.032171    1.962906    0.642553    1.000000      0.357447         0.526646  00:16     
1         1.005825    0.782014    0.565957    0.416667      0.294118         0.344828  00:16     
2         1.007390    0.853155    0.600000    0.797619      0.350785         0.487273  00:16     
3         0.955742    2.763814    0.642553    1.000000      0.357447         0.526646  00:16     
4         0.940904    4.990465    0.642553    1.000000      0.357447         0.526646  00:17     
5         0.918661    3.840919    0.642553    1.000000      0.357447         0.526646  00:16     
6         0.885470    3.328572    0.642553    1.000000      0.357447         0.526646  00:16     
7         0.845819    2.020619    0.642553    1.000000      0.357447         0.526646  00:16     
8         0.819089    0.913207    0.651064    0.845238      0.336493         0.481356  00:17     
9         0.770136    0.671836    0.387234    0.166667      0.400000         0.235294  00:16     
10        0.732953    0.659600    0.357447    0.142857      0.500000         0.222222  00:16     
11        0.706902    0.683854    0.353191    0.023810      0.666667         0.045977  00:17     
12        0.678196    0.682978    0.344681    0.047619      0.800000         0.089888  00:16     
13        0.652133    0.688035    0.353191    0.023810      0.666667         0.045977  00:16     
14        0.629146    0.684688    0.353191    0.071429      0.545455         0.126316  00:16   

When the batch size is 4, at epoch 6 we get a low error rate of 0.327660 (I just noticed this now so I will have to add this to a hyperparameter I have to explore). We notice that for batch size 32, we get a low error rate of ~0.36 and comparatively low validation and training loss at epoch 12. For batch size 16, we get a low error rate of ~0.37 and comparatively low validation and training loss at epoch 10. Batch size 64 also gets low error rates of around 0.34-0.39 after epoch 9 with relatively low training and validattion losses.

When training with a batch size 100, the terminal returns a cuda out of memory error so we have to resize (reduce) the size of the images.

#### Resizing Testing:
new approach resize down 400x400, batch size 128, 

Resnet34, Batch size  128, epoch 15, resize factor 400
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.206917    0.728770    0.357447    0.178571      0.500000         0.263158  00:15     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.129427    0.736311    0.374468    0.416667      0.472973         0.443038  00:18     
1         1.115820    1.192699    0.608511    0.904762      0.360190         0.515254  00:18     
2         1.081827    1.067438    0.361702    0.059524      0.454545         0.105263  00:18     
3         1.040130    1.521363    0.357447    0.000000      0.000000         0.000000  00:18     
4         0.992024    0.843747    0.497872    0.583333      0.374046         0.455814  00:18     
5         0.971873    1.653695    0.638298    0.976190      0.356522         0.522293  00:18     
6         0.931452    1.143807    0.625532    0.904762      0.353488         0.508361  00:18     
7         0.896557    1.024012    0.629787    0.904762      0.351852         0.506667  00:18     
8         0.864508    1.441459    0.642553    1.000000      0.357447         0.526646  00:18     
9         0.832286    2.022151    0.642553    1.000000      0.357447         0.526646  00:18     
10        0.810683    2.851242    0.642553    1.000000      0.357447         0.526646  00:18     
11        0.785335    2.860525    0.642553    1.000000      0.357447         0.526646  00:18     
12        0.763332    2.928426    0.642553    1.000000      0.357447         0.526646  00:18     
13        0.743765    2.998181    0.642553    1.000000      0.357447         0.526646  00:18     
14        0.733455    3.007145    0.642553    1.000000      0.357447         0.526646  00:18     

double test
Resnet34, Batch size  128, epoch 15, resize factor 400
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.206917    0.728770    0.357447    0.178571      0.500000         0.263158  00:15     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.129427    0.736311    0.374468    0.416667      0.472973         0.443038  00:18     
1         1.115820    1.192699    0.608511    0.904762      0.360190         0.515254  00:18     
2         1.081827    1.067438    0.361702    0.059524      0.454545         0.105263  00:18     
3         1.040130    1.521363    0.357447    0.000000      0.000000         0.000000  00:18     
4         0.992024    0.843747    0.497872    0.583333      0.374046         0.455814  00:18     
5         0.971873    1.653695    0.638298    0.976190      0.356522         0.522293  00:18     
6         0.931452    1.143807    0.625532    0.904762      0.353488         0.508361  00:18     
7         0.896557    1.024012    0.629787    0.904762      0.351852         0.506667  00:18     
8         0.864508    1.441459    0.642553    1.000000      0.357447         0.526646  00:18     
9         0.832286    2.022151    0.642553    1.000000      0.357447         0.526646  00:18     
10        0.810683    2.851242    0.642553    1.000000      0.357447         0.526646  00:18     
11        0.785335    2.860525    0.642553    1.000000      0.357447         0.526646  00:18     
12        0.763332    2.928426    0.642553    1.000000      0.357447         0.526646  00:18     
13        0.743765    2.998181    0.642553    1.000000      0.357447         0.526646  00:18     
14        0.733455    3.007145    0.642553    1.000000      0.357447         0.526646  00:18  

increase resizing to see what happens to 500

Resnet34, Batch size  128, epoch 15, resize factor 500
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.206917    0.728770    0.357447    0.178571      0.500000         0.263158  00:14     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.129427    0.736311    0.374468    0.416667      0.472973         0.443038  00:18     
1         1.115820    1.192699    0.608511    0.904762      0.360190         0.515254  00:18     
2         1.081827    1.067438    0.361702    0.059524      0.454545         0.105263  00:18     
3         1.040130    1.521363    0.357447    0.000000      0.000000         0.000000  00:18     
4         0.992024    0.843747    0.497872    0.583333      0.374046         0.455814  00:18     
5         0.971873    1.653695    0.638298    0.976190      0.356522         0.522293  00:18     
6         0.931452    1.143807    0.625532    0.904762      0.353488         0.508361  00:18     
7         0.896557    1.024012    0.629787    0.904762      0.351852         0.506667  00:18     
8         0.864508    1.441459    0.642553    1.000000      0.357447         0.526646  00:18     
9         0.832286    2.022151    0.642553    1.000000      0.357447         0.526646  00:18     
10        0.810683    2.851242    0.642553    1.000000      0.357447         0.526646  00:18     
11        0.785335    2.860525    0.642553    1.000000      0.357447         0.526646  00:18     
12        0.763332    2.928426    0.642553    1.000000      0.357447         0.526646  00:18     
13        0.743765    2.998181    0.642553    1.000000      0.357447         0.526646  00:18     
14        0.733455    3.007145    0.642553    1.000000      0.357447         0.526646  00:18    

resizing factor 300

Resnet34, Batch size  128, epoch 15, resize factor 300
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.206917    0.728770    0.357447    0.178571      0.500000         0.263158  00:15     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         1.129427    0.736311    0.374468    0.416667      0.472973         0.443038  00:18     
1         1.115820    1.192699    0.608511    0.904762      0.360190         0.515254  00:18     
2         1.081827    1.067438    0.361702    0.059524      0.454545         0.105263  00:18   
3         1.040130    1.521363    0.357447    0.000000      0.000000         0.000000  00:18     
4         0.992024    0.843747    0.497872    0.583333      0.374046         0.455814  00:18     
5         0.971873    1.653695    0.638298    0.976190      0.356522         0.522293  00:18     
6         0.931452    1.143807    0.625532    0.904762      0.353488         0.508361  00:18     
7         0.896557    1.024012    0.629787    0.904762      0.351852         0.506667  00:18     
8         0.864508    1.441459    0.642553    1.000000      0.357447         0.526646  00:18     
9         0.832286    2.022151    0.642553    1.000000      0.357447         0.526646  00:18     
10        0.810683    2.851242    0.642553    1.000000      0.357447         0.526646  00:18     
11        0.785335    2.860525    0.642553    1.000000      0.357447         0.526646  00:18     
12        0.763332    2.928426    0.642553    1.000000      0.357447         0.526646  00:18     
13        0.743765    2.998181    0.642553    1.000000      0.357447         0.526646  00:18     
14        0.733455    3.007145    0.642553    1.000000      0.357447         0.526646  00:18     

We notice here that resizing and reducing the size of the images does not lead to information loss thus the results are not impacted when the quality of the images is reduced.

#### Final Hyperparameter Trials:

Resnet34, Batch size  32, epoch 15
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.592978    0.556228    0.248454    0.000922      0.500000         0.001840  03:32     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
/home/CAMPUS/slab2019/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
0         0.488386    0.615993    0.248454    0.000000      0.000000         0.000000  04:50     
1         0.419885    0.575407    0.248454    0.000000      0.000000         0.000000  04:52     
2         0.418290    0.546055    0.248454    0.000000      0.000000         0.000000  04:50     
3         0.400961    0.569583    0.217312    0.435023      0.584158         0.498679  04:51     
4         0.373029    0.643854    0.374628    0.808295      0.380477         0.517404  04:49     
5         0.377766    0.554076    0.256011    0.664516      0.488814         0.563281  04:48     
6         0.355055    0.449868    0.192810    0.444240      0.668516         0.533776  04:48     
7         0.325130    0.457167    0.209755    0.606452      0.573670         0.589606  04:48     
8         0.303448    0.522989    0.247996    0.705069      0.500654         0.585534  04:47     
9         0.315929    0.647850    0.333181    0.811982      0.413227         0.547715  04:48     
10        0.301271    0.620199    0.316693    0.775115      0.424747         0.548777  04:47     
11        0.255606    0.570673    0.277765    0.688479      0.460543         0.551902  04:47     
12        0.265308    0.656293    0.339592    0.787097      0.405508         0.535255  04:47     
13        0.259017    0.608794    0.296084    0.752074      0.443478         0.557949  04:47     
14        0.270167    0.690407    0.344401    0.787097      0.401504         0.531756  04:47  

Resnet34, Batch size  64, epoch 15
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.677108    0.636303    0.248454    0.000000      0.000000         0.000000  03:23     
epoch     train_loss  valid_loss  error_rate  recall_score  precision_score  f1_score  time    
0         0.488871    0.698323    0.248454    0.000000      0.000000         0.000000  04:39     
1         0.436395    0.600500    0.248454    0.000000      0.000000         0.000000  04:39     
2         0.417207    0.560733    0.248683    0.000922      0.333333         0.001838  04:39     
3         0.394394    1.140321    0.750401    1.000000      0.248739         0.398384  04:39     
4         0.378545    0.726919    0.670025    0.949309      0.264035         0.413157  04:39     
5         0.356737    0.701254    0.508816    0.863594      0.311192         0.457520  04:39     
6         0.347170    1.368876    0.751088    1.000000      0.248568         0.398165  04:39     
7         0.327890    0.850209    0.663613    0.972350      0.268927         0.421326  04:39     
8         0.314793    0.946016    0.746737    0.997235      0.249309         0.398894  04:39     
9         0.310391    0.908477    0.689947    0.971429      0.261150         0.411638  04:39     
10        0.284364    1.149254    0.746508    0.994470      0.249019         0.398302  04:39     
11        0.257435    1.723110    0.750630    0.998157      0.248451         0.397869  04:39     
12        0.261871    1.682301    0.749714    0.997235      0.248564         0.397940  04:39     
13        0.248655    2.259446    0.750630    0.999078      0.248567         0.398090  04:39     
14        0.240725    2.521528    0.750630    1.000000      0.248682         0.398311  04:39

// loss a bs 16 and bs 4 trial run because my VSCode Window crashed and I couldn't ssh back into the terminal, will include it after I run it again. sorry :'(.
Because of this we cannot form a conclusion of what is the best set of hyperparameters from the given sets mentioned in the methods section. However, we can note that when training the model with the entire dataset and a batch size of 64, we can achieve an error rate of 0.248454 from epoch 0 to epoch 2. The error rate for batch size 32 is the same from epoch 0 to epoch 2, but the training and validation loss of batch size 32 is marginally lower and thus better. We can reduce the epochs from 15 to 3 now that we know the error rate gets significantly worst after 3 epochs.

With our first generated image dataset which was 512x512 images, we could not increase the batch size beyond 16. In order to pass this threshold, we reduced the size of the images in the second data set and experimented with different image sizes and various batch sizes.
Below are the results from our batch size testing on reduced image sizes:

Control: Batch size = 16, no resizing
epoch     train_loss  valid_loss  error_rate  time    
0         1.193201    0.861575    0.305000    00:07                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.175750    0.778355    0.350000    00:10                              
1         1.103686    0.751094    0.300000    00:10                              
2         0.972977    0.609927    0.260000    00:10                              
3         0.879336    0.703814    0.360000    00:10 

Batch size = 16, items_tfms=Resize(300)
epoch     train_loss  valid_loss  error_rate  time    
0         1.193201    0.861575    0.305000    00:08                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.175750    0.778355    0.350000    00:10                              
1         1.103686    0.751094    0.300000    00:10                              
2         0.972977    0.609927    0.260000    00:10                              
3         0.879336    0.703814    0.360000    00:10 

Batch size = 16, items_tfms=Resize(300)
epoch     train_loss  valid_loss  error_rate  time    
0         1.193201    0.861575    0.305000    00:07                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.175750    0.778355    0.350000    00:10                              
1         1.103686    0.751094    0.300000    00:10                              
2         0.972977    0.609927    0.260000    00:09                              
3         0.879336    0.703814    0.360000    00:09 

Batch size = 14
epoch     train_loss  valid_loss  error_rate  time    
0         1.174931    0.863961    0.335000    00:08                                                                 
epoch     train_loss  valid_loss  error_rate  time    
0         1.166424    1.126207    0.455000    00:10                                                                 
1         1.031477    0.636147    0.260000    00:10                                                                 
2         0.954057    0.958069    0.420000    00:10                                                                 
3         0.815557    0.648650    0.300000    00:10

Batch size = 12
epoch     train_loss  valid_loss  error_rate  time    
0         1.204409    0.958627    0.400000    00:08                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.151761    1.094397    0.395000    00:10                              
1         1.023440    2.890649    0.495000    00:10                              
2         0.891081    0.844348    0.360000    00:10                              
3         0.826468    0.630925    0.310000    00:10 


Batch size = 24, items_tfms=Resize(400)
epoch     train_loss  valid_loss  error_rate  time    
0         1.167504    0.818557    0.345000    00:07                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.117804    0.810464    0.335000    00:09                              
1         1.038550    0.655865    0.275000    00:09                              
2         0.987740    0.615587    0.275000    00:09                              
3         0.919624    0.536605    0.275000    00:09  

Batch size = 32, items_tfms=Resize(400)
epoch     train_loss  valid_loss  error_rate  time    
0         1.128878    0.863806    0.370000    00:07                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.112206    0.707882    0.355000    00:09                              
1         1.031291    0.644120    0.300000    00:09                              
2         0.947920    0.712319    0.300000    00:09                              
3         0.867006    0.690136    0.285000    00:09

Batch size = 32, items_tfms=Resize(400)
epoch     train_loss  valid_loss  error_rate  time    
0         1.179415    0.832696    0.415000    00:07                              
epoch     train_loss  valid_loss  error_rate  time    
0         1.124885    0.746533    0.350000    00:09                              
1         1.061206    0.977342    0.370000    00:09                              
2         1.014017    0.754363    0.315000    00:09                              
3         0.958239    0.658920    0.285000    00:09  

As you can see, increasing the batch size with 400x400 images did not lead to improvements in the error_rate (from 0.285 error rate in epoch 3 batch_size=24 to 0.285 error rate in epoch 3 batch_size=32). Our testing needs to further check the increase from a batch size of 16 to a batch size of 24 for 400x400 images. 

## RNN Testing
The RNN we created was unsuccessful: we observed that throughout the entirety of the training process, the cost fluctuated drastically from training sample to training sample, rotating among the values of 0.0, 50.0 and 100.0. Since the training cost exhibited no decreasing or stabilizing patterns, we conclude that our RNN failed to learn.

## Overall results 

Our CNN results suggest the feasibility of image-based EKG classification, although also pointing to the need to augment transfer learning with problem-specific techniques in future work.  While our classifier achieves a high rate of accuracy, its sensitivity is not yet high enough for clinical application, where the acceptable error rate is very low.  Missing an MI could result in lack of treatment that could potentially cause outcomes up to death, while false positive diagnosis could potentially subject a patient to unnecessary invasive procedures with similarly severe consequences.  Error rates in STEMi classification by Emergency Room Physicians are approximately 3% [\[5\]](#Citations), and prior non-image based work has achieved similar levels of accuracy.  Techniques from this prior work could potentially be used to improve our own results,  such as lead pooling and sub 2D convolutional layers[\[7\]](#Citations), and pre-processing images for noise reduction and pulse segmentation[\[8\]](#Citations).  However, our classifier already substantially outperforms general physicians and Emergency Room Residents, who have been found to have accuracy rates of 70% and 68% respectively [\[6\]](#Citations).  It also outperforms the non-neural network based algorithm used in the LifePak 12, one of the most popular devices for pre-hospital 12-lead EKG acquisition by EMS, which one study noted only detected 58% of STEMIs [\[4\]](#Citations).

## Ethics
Our work touches on a number of ethical issues and dilemmas, including: 
- **Who is held responsible in the case where the model's prediction is incorrect and the diagnosis and treatment lead to an unfavorable outcome for the patient?**
   - We imagine that our model would be used in situations where the caregiver is not properly trained to read EKGs, so the patient is already put at risk of inaccurate diagnosis. In this specific case, it seems like the neither the model, nor the caregiver should be held accountable, simply because the latter is not trained to read an EKG and hence respond appropriately already, so the addition of our model would technically only improve the rates of accurate and timely diagnosis. If there's ever an implementation used by medical professionals trained to read EKGs, it seems like they should be the ones responsible, as they are making the final call. In general, our model only analyzes the EKG as accurately as possible, but is not the entity reponsible for the actions taken given this information. To avoid such situations as much as possible, NN models should be implemented in the real world only after certain accuracy is achieved, so that they don't lead to loss of life and other negative consequences. In any situation where the technology is the one that makes the final decision, the developers and manufacturer/company should be held accountable.
- **Issues around collection of sensitive health data - should it be collected in an effort to improve the model? In what way should it be collected?**
  - Data from the real world would be essential in improving the accuracy of the model. In that sense, it would be helpful to collect such data, even though it is highly sensitive. In order to do that in a more sensitive way, this information should be anonymized and consent from the patient should be obtained (once they are able to give it). 
- **Does the data used to train the model capture the heterogeneity of the real world? How can a more comprehensive dataset be assembled?**
  - Most likely the datset we're currently using is not incredibly diverse. A potential way to address this issue is to partner with hospitals and medical centers around the world to collect data from a diverse group of individuals. This approach introduces several complications, such as what incentive would such institutions have to collect and provide that data; having to navigate different health systems and ways that consent is obtained (would consent obtained in, for example, Vietnam be accepted in wherever the company is situated/headquartered?); and other logistical questions.
- **How do we find and remove potential biases in the dataset?** 
  - Similarly to the previous question, we need to make sure that we are gathering data from different parts of the world instead of just one hospital or just one country. Similarly, finding potential biases would require testing the model on a diverse range of data.
- **Who benefits from the development and potential commercialization of such a tool?**
  - The main beneficiary (at least financially) would definitely be the company that owns the product. The introduction of such a technology aims at improving the rates of correct and timely diagnoses and consequently lower the rates of improper treatment or even death. Therefore, a model that is able to accurately read EKGs would also benefit health workers and patients. Additionally, if the model reaches a high accuracy rate, it may even be beneficial for smaller or under-funded hospitals and medical institutions, as it would allow a quick and relatively cheap solution to diagnosing patients. This raises other important ethical questions about technology replacing humans and taking over what could potentially be jobs performed by people.

Addressing these questions in depth is important in assessing the model and making sure that introducing it into the health system will be beneficial and not cause harm.

# Reflection

Our work mostly acts as a proof of concept, pointing to the possibility of future work by researchers with access to proprietary EKG image datasets and/or partnership with clinical researchers to confirm the viability of classifying EKG images obtained in real clinical settings. While our goal was to create an app that assists EMTs and other health professionals in quickly and accurately diagnosing patients, we recognize that our current model requires more work and calibration to achieve this. For example, the image dataset that we generated is still not sufficiently similar to real EKGs, or what a picture of an EKG may look like. While we tried to plot and visualize the data to resemble as closely as possible an actual EKG, improvements can be made. For example, the plot lines can be made to appear red, which is typically the color of EKGs (see Fig. 2); further image augmentation and more realistic shadows may be implemented to mimic real-world pictures; in general, improvements in the visual representation of the data that make it appear more similar to real EKG printouts may lead to lower error rates when the model is tested on real pictures. Ideally, a dataset of real pictures of EKGs can be assembled and used to train the model.

Additionally, further experimentation with adding more and different types of layers to the NN, as well as changing batch size, is needed to achieve the best accuracy-computation cost trade-off. Finally, in an attempt to asses the need for such a tool (as well as its impact), a more thorough understanding of the ethics must be achieved.




## Citations
1. [Virani SS, Alonso A, Aparicio HJ, Benjamin EJ, Bittencourt MS, Callaway CW, Carson AP, Chamberlain AM, Cheng S, Delling FN, Elkind MSV, Evenson KR, Ferguson JF, Gupta DK, Khan SS, Kissela BM, Knutson KL, Lee CD, Lewis TT, Liu J, Loop MS, Lutsey PL, Ma J, Mackey J, Martin SS, Matchar DB, Mussolino ME, Navaneethan SD, Perak AM, Roth GA, Samad Z, Satou GM, Schroeder EB, Shah SH, Shay CM, Stokes A, VanWagner LB, Wang N-Y, Tsao CW; on behalf of the American Heart Association Council on Epidemiology and Prevention Statistics Committee and Stroke Statistics Subcommittee. Heart disease and stroke statistics—2021 update: a report from the American Heart Association [published online ahead of print January 27, 2021]. Circulation.doi: 10.1161/CIR.0000000000000950](https://www.heart.org/-/media/phd-files-2/science-news/2/2021-heart-and-stroke-stat-update/2021_heart_disease_and_stroke_statistics_update_fact_sheet_at_a_glance.pdf?la=en)
2. [Bång, Angela, Lars Grip, Johan Herlitz, Stefan Kihlgren, Thomas Karlsson, Kenneth Caidahl, and Marianne Hartford. "Lower mortality after prehospital recognition and treatment followed by fast tracking to coronary care compared with admittance via emergency department in patients with ST-elevation myocardial infarction." International journal of cardiology 129, no. 3 (2008): 325-332.](https://www.sciencedirect.com/science/article/pii/S0167527307016579?casa_token=QwM9I3I5klIAAAAA:1BTMwOBPmN4yl27K4MK_dxenVVpPWVXrzWEmp2Sid99Vjj-018TLvvhR7CRVz5MGYgCmvs4a_A)
3. [Turner, A., Dunne, R. and Wise, K., 2017. National Institute For Health Care Reform. [online] Nihcr.org. Available at: <https://nihcr.org/wp-content/uploads/2017/06/NIHCR_Altarum_Detroit_EMS_Brief_5-30-17.pdf> [Accessed 8 May 2021].](https://nihcr.org/wp-content/uploads/2017/06/NIHCR_Altarum_Detroit_EMS_Brief_5-30-17.pdf)
4. [ Mary Colleen Bhalla, Francis Mencl, Mikki Amber Gist, Scott Wilber & Jon Zalewski (2013) Prehospital Electrocardiographic Computer Identification of ST-segment Elevation Myocardial Infarction, Prehospital Emergency Care, 17:2, 211-216, DOI: 10.3109/10903127.2012.722176 ](https://www.tandfonline.com/doi/abs/10.3109/10903127.2012.722176)
5. [ Hartman, Stephanie M., Andrew J. Barros, and William J. Brady. "The use of a 4-step algorithm in the electrocardiographic diagnosis of ST-segment elevation myocardial infarction by novice interpreters." The American journal of emergency medicine 30, no. 7 (2012): 1282-1295. ](https://emupdates.com/wp-content/uploads/2008/07/Hartman-4-Steps-to-STEMI-Diagnosis-AmJEM-2012.pdf)
6. [ Mehta, S., F. Fernandez, C. Villagran, A. Frauenfelder, C. Matheus, D. Vieira, M. A. Torres et al. "P1466 Can physicians trust a machine learning algorithm to diagnose ST elevation myocardial infarction?." European Heart Journal 40, no. Supplement_1 (2019): ehz748-0231. ](https://academic.oup.com/eurheartj/article-abstract/40/Supplement_1/ehz748.0231/5598215)
7. [ Liu, Wenhan, Mengxin Zhang, Yidan Zhang, Yuan Liao, Qijun Huang, Sheng Chang, Hao Wang, and Jin He. "Real-time multilead convolutional neural network for myocardial infarction detection." IEEE journal of biomedical and health informatics 22, no. 5 (2017): 1434-1444.](https://ieeexplore.ieee.org/document/8103330)
8. [ Park, Yeonghyeon, Il Dong Yun, and Si-Hyuck Kang. "Preprocessing method for performance enhancement in cnn-based stemi detection from 12-lead ecg." IEEE Access 7 (2019): 99964-99977. ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8771175)
9. [Xiao, Ran, Yuan Xu, Michele M. Pelter, David W. Mortara, and Xiao Hu. "A deep learning approach to examine ischemic ST changes in ambulatory ECG recordings." AMIA Summits on Translational Science Proceedings 2018 (2018): 256.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961830/ )
10. [ Al-Zaiti, Salah, Lucas Besomi, Zeineb Bouzid, Ziad Faramand, Stephanie Frisch, Christian Martin-Gill, Richard Gregg, Samir Saba, Clifton Callaway, and Ervin Sejdić. "Machine learning-based prediction of acute coronary syndrome using only the pre-hospital 12-lead electrocardiogram." Nature communications 11, no. 1 (2020): 1-10.](https://www.nature.com/articles/s41467-020-17804-2)
11. [ Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.1). PhysioNet. https://doi.org/10.13026/x4td-x982. ](https://www.physionet.org/content/ptb-xl/1.0.1/)
12. [ Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data. https://doi.org/10.1038/s41597-020-0495-6 ] (https://www.nature.com/articles/s41597-020-0495-6)
13. [ Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. ](https://www.physionet.org/content/ptb-xl/1.0.1/)

