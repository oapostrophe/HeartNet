# Introduction

Heart disease is consistently the number one cause of death in America, and 805,000 heart attacks occur in the US annually. (1)  Effective heart attack treatment relies on rapid diagnosis through 12-lead EKG in order to minimize the time to reperfusion therapy, and one of the largest improvements in the speed of diagnosis and treatment has been achieved through pre-hospital EKG analysis by Emergency Medical Services personnel.  For instance, one study found pre-arrival EMS identification of ST-Elevated Myocardial Infarction through 12-lead EKG lowered patients’ 30-day mortality rates from 15.3% to 7.3%, and 5-year mortality rates from 20.6% to 11.6%. (2)

Yet despite the well-established lifesaving effects of performing pre-hospital 12-lead EKGs, many EMS systems continue to rely on Basic Life Support-level providers untrained in EKG interpretation.  For instance, the Detroit EMS system, which handles approximately 100,000 calls annually is predominately staffed by BLS units, and Michigan has a statewide shortage of hundreds of ALS-level paramedics who would be trained to perform EKGs. (3)  The result is that heart attack patients don't have EKGs performed before hospital arrival, resulting in slower diagnosis, slower treatment, and substantially higher mortality.

In such EMS systems where staff trained in EKG interpretation are unavailable, automatic computer interpretation of EKG could potentially fill in the gap to provide the lifesaving benefits of pre-hospital diagnosis to  heart attack patients.  However, current STEMI detection algorithms in use on EKG monitors are famously inaccurate and unreliable.  For instance, one study found that a commonly used EMS monitor identified just 58% of STEMis, potentially resulting in 42/100 missed cath lab activations. (4)

The past few years have seen researchers achieve breakthrough accuracy levels in automatic EKG interpretation using neural-network based approaches, in many cases rivaling or surpassing the accuracy of clinical providers.  However, such algorithms have yet to be widely applied to actual healthcare practice.  One barrier to such application has been the fact that all such algorithms have been developed to interpret digital EKG signals, which in the field are only available to software directly interfacing with an EKG monitor.  Clinical implementation of such algorithms would either need to be done on proprietary software run on EKG monitors themselves, or else on specialized hardware with the ability to interface directly with EKG monitors to receive digital signals.  In either case, such implementation would be difficult both for researchers and clinical practitioners.

We attempt to resolve this difficulty by testing an alternate approach to neural-network based STEMi detection using images.  While digital signals are difficult to directly obtain in clinical practice, EKG monitors are equipped with printing capabilities that are routinely used to record readings.  If neural networks can accurately detect STEMis in images of 12-lead EKGs, then such printouts could be used as the basis of classification independently of the specific EKG device.  For instance, we envision an image-based approach being the basis of a mobile phone application that would allow providers to easily take a photo of an EKG printout and have it automatically classified.   Such an application could be developed by any researchers without needing access to proprietary software, and could be applied in clinical practice by healthcare agencies without needing to purchase specialized equipment.  Thus the feasibility of an image-based classification approach for EKGs could represent a large step in bridging the gap between research achievements and clinical practice.

A challenge to this approach, however, is the lack of a publicly available dataset of annotated 12-lead EKG printouts.  To test the feasibility of our approach with publicily available data, we create a dataset by plotting each lead of digital signal-based readings and combining the plots from each lead into a single image.  The result is images that approximate 12-lead EKG printouts.

Another substantial challenge is the fact that, depending on the part of the heart affected by MI, diagnostic features such as ST-elevation can appear in different leads and thus different parts of the image.  This challenge requires us to design and train our neural network in a way that can recognize these features regardless of where they appear in the image.  Furthermore, even baseline EKGs can look dramatically different from patient to patient with a wide variety of arrhythmias existing, and MI potentially presenting with different characteristic features.  We attempt to overcome this fact by training on a large and diverse dataset, as well as carefully designing the layers of our network.

Our finished neural network replicates the high level of accuracy seen in previous work, suggesting the feasibility of automatically classifying EKGs through images of printouts.  These results support the development of an application for computer STEMi recognition that can be easily integrated into clinical practice, and suggest that such an application can extend the lifesaving benefits of pre-hospital EKG interpretation to patients currently unable to receive it.

Ethical concerns for our project include potential biases in our dataset, which could influence our model’s performance.  Testing an application based on our model in clinical practice will also raise ethical concerns due to not only concerns for patient safety, but the difficulty of obtaining informed and carefully considered consent from patients suffering from heart attack.  Finally, the use of neural networks to make clinical treatment decisions raises broader concerns about the influence such practices may have on healthcare system design and practice.

# Related Works:

In this section, we will discuss relevant research pertaining to the application of neural networks to STEMi detection in EKGs. Past studies using convolutional neural networks (CNNs) have achieved high accuracy and sensitivity classifying STEMi's from various sets of leads. A study using 4-lead EKGs, sub-2d convolutional layers, lead asymmetric pooling layers to combine data from the multiple leads to achieve a 96% accuracy classifying Anterior STEMis. The algorithm's real-time performance was tested on a lightweight embedded system and found to be suitable for application in EKG wearable technologies. (Liu, et al., 2018)

A study more pertinent to our project's scope combines image-based deep learniing techniques to improve detection accuracy of an important marker for detecting myocardial ischemia in EKGs: a ST depression change. The CNN created yields an average AUC at 89.6% from an independent set, a mean sensitivity rate at 84.4% and a specificity at 84.9% at selected optimal cutoff thresholds. (Xiao, et al., 2018)

Another study using a CNN and 12-lead EKGs, proposed a performance optimization technique through two data pre-processing methods: noise reduction (notch filter and high pass filter) and pulse segmentation (via QRS complex detection). The 
preprocessing techniques improved the sensitivity, specificty, and area under the curve (AUC) of the receiver operatnig characteristic (ROC), enhanncing STEMi detection performance on a 275 EKG record dataset with 179 STEMis and 96 normal (Park, et al., 2019).

Due to the complexity of classifiers like CNNs and other neural networks, key decision makers like physicians and experienced clinicians stigmatize the black-box nature of neural network-based diagnoses. A more recent study using a ML fusion model consistening of Logistic Regression (LR), Artificial Neural Network (ANN), and Gradient Boosting Machine (GBM) and modified approach using 554 temporal-spatial features of 12-lead EKGs from a sample size of 1244 patients was able to achieve a 52% gain over commercial software ad 37% gain over "experienced" clinicians. From the study, the researchers concluded that linear classifiers like LR are just as effect as ANN, which lends the use of linear classification favorability in clinical practice (Al-Zaiti, et al., 2020).

# Methods

A dataset of labelled EKG images was generated ("MI", for "Myocardial Infraction," or "NORM," for "Normal/Healthy") . The images were made by plotting individual leads in matplotlib and appending them together. The labels were extracted from a CSV document, containing the raw EKG data from PTB-XL, a large publicly available electrocardiography dataset. Subsequently, OpenCV was used to transform the plots to 512x512 pixel grayscale images and append them into a 3x4 image for each patient, similar to how an EKG printout is organized. Initially, sample images were generated for a set of patients; eventually, images depicting the EKG data were generated for all patients in the dataset. 
In an effort to improve the quality of the generated images, WFDB library's plot function was used to make them appear smoother and resemble more closely real EKGs.

The software used to build and train the neural network was PyTorch and fastai. 

As of right now, we are not sure what tools we will be using for analysis, but we will look into whether fastai offers any good ways to do that, as well as research other options, such as statistical packages like scipy.stats. We would want to check how many heart attacks the neural network misses to identify, which is different than overall accuracy, and how many false positives come up.

# Discussion

Our first generated image dataset successfully plotted EKG waveforms and allowed training of a CNN; however, the generated images bore several striking differences from real-world EKG printouts. 

Our second generated dataset eliminated most of these differences and resulted in images appearing much closer to real EKG printouts. 

We then experimented with data augmentation by using FastAI to introduce artifact such as shadow that might appear on photographs of EKG printouts. 

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

Our results suggest the feasibility of image-based EKG classification in clinical practice, although also point to the need for future work to augment transfer learning with other problem-specific techniques to attain accuracy high enough for clinical application, where the acceptable error rate is very low. 

Our work mostly acts as a proof of concept, pointing to the possibility of future work by researchers with access to proprietary EKG image datasets and/or partnership with clinical researchers to confirm the viability of classifying EKG images obtained in real clinical settings.


## Citations:
1. https://www.heart.org/-/media/phd-files-2/science-news/2/2021-heart-and-stroke-stat-update/2021_heart_disease_and_stroke_statistics_update_fact_sheet_at_a_glance.pdf?la=en
2. Bång A, Grip L, Herlitz J, et al. Lower mortality after prehospitalrecognition and treatment followed by fast tracking to coronary carecompared with admittance via emergency department in patients withST-elevation myocardial infarction. Int J Cardiol 2008;129:325-32
3. https://nihcr.org/wp-content/uploads/2017/06/NIHCR_Altarum_Detroit_EMS_Brief_5-30-17.pdf
4. https://www.tandfonline.com/doi/abs/10.3109/10903127.2012.722176
