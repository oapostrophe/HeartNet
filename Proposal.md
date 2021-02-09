# HeartNet: an app for neural network EKG interpretation


![EKG image](/images/ekg.jpg)
(A picture of an EKG indicating an ST-Elevated Myocardial Infarction, more commonly known as a heart attack)

## Project Description

Electrocardiograms, or EKGs, are a clinical test that records a visual pattern of the heart’s electrical activity.  The shape of this pattern can be interpreted by trained medical providers to spot signs of a heart attack and other heart conditions.  Historically, attempts to automatically interpret these waveforms with computer algorithms have produced poor results.  In the past few years,  however, researchers have been able to achieve groundbreaking accuracy levels using neural networks, even creating programs that interpret EKGs more accurately than cardiologists.

![12-lead image](/images/12lead.jpg)
  
There are a number of healthcare settings in which automatic EKG interpretation could save patients’ lives.  For instance, many big-city ambulance services respond to most cardiac emergencies with technicians not trained in EKG interpretation.  This results in delayed treatment for conditions like heart attacks that go undetected until the patient arrives at the ER.  What if these EMTs could do an EKG, take a picture with their phone and have a neural network instantly alert them that the patient is having a heart attack?  Such a system could be implemented by EMS systems without hiring new personnel, developing new medical devices, and in many cases without even needing to purchase new equipment.  Yet despite the accurate models created by researchers, such applications do not exist.
  
I’d propose to create a neural network based application that makes accurate neural network-based EKG interpretation accessible to the general public.  Building on prior models, I want to create either a website or mobile phone app where users can take a picture of an EKG and quickly get an accurate classification.  Such an application could be a proof of concept for how healthcare providers, particularly EMS medical directors, could easily incorporate neural-network based EKG interpretation into their medical systems.

## Project Goals

1. Accurately detect STEMis in images of 12-lead EKGs
2. Classify a range of general heart rhythms and arrhythmias
3. Test accuracy of prior models based on clean images / digital readings when used on photos possibly containing shadow and other artifact
4. Consider accuracy with pre-hospital EKGs that may have more noise or other artifact in waveform
5. Possibly build a new dataset and model based on print outs of EKG images from existing datasets.
