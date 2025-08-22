# AI-Driven Sleep Staging Using Instantaneous Heart Rate and Accelerometry: Insights from an Apple Watch Study
Tzu-An Song, Yubo Zhang, Ziyuan Zhou, Luke Hou, Masoud Malekzadeh, Aida Behzad, and Joyita Dutta

Abstract—Polysomnography, the gold standard for sleep evaluations,
involves complex setup and data acquisition protocols and
requires manual scoring of sleep data. Smartwatches and other
multi-sensor consumer wearable devices with automated sleep
staging capabilities offer a promising and scalable alternative for
routine and long-term sleep evaluations in individuals. We conducted
a multi-night study using a smartwatch for sleep assessment
and created an AI-driven automated sleep staging framework
based on instantaneous heart rate (IHR) and accelerometry data
using sleep stage labels based on electroencephalography (EEG)
as the reference. 47 healthy adults were recruited to record their
sleep for up to seven consecutive nights using an Apple Watch
Series 6 and a Dreem 2 Headband. Our sleep staging framework
relies on a sequence-to-sequence long short-term memory (LSTM)
model with additional convolutional layers. Our model yields a
sleep staging accuracy of 71% for classifying every 30-s epoch
into four classes: wake, light sleep, deep sleep, and rapid eye
movement (REM) sleep. We show through an ablation study that
an intra-epoch learning LSTM, incorporation of IHR sampling frequency
information, and skip connections from early to late stages
of the network are three key architectural advancements that enhance
overall sleep staging performance. Our overall contributions
include a dedicated Apple Watch app for multi-night raw data
acquisition, an open-source library for automated four-class sleep
staging, and a public dataset for future investigations.
## Prerequisites

This code uses:

- Python 2.7
- Pytorch 0.4.0
- matplotlib 2.2.4
- numpy 1.16.4
- scipy 1.2.1
- NVIDIA GPU
- CUDA 8.0
- CuDNN 7.1.2
## Dataset
The Apple Watch dataset will be released soon.
## UMASS_Amherst_BIDSLab
Biomedical Imaging & Data Science Laboratory

Lab's website:
http://www.bidslab.org/index.html


Email: bidslab(at)gmail.com,
       tzuansong(at)umass.edu.
