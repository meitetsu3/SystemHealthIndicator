# System Health Indicator with Auto-Encoder



## Overview

The use of deep auto-encoder [2] to create system health indicator in unsupervised manner along with fault diagnostics information seems very beneficial. Application of the method to many different field may
help understand the robustness and applicability of the approach. In this research, the auto-encoder described in [2] is implemented and its health indicator and diagnostic capability is evaluated using the vibration data described in [1]. 

### Datasets and Inputs

 The data is available from [CaseWestern Reserve University Bearing Data Center](http://csegroups.case.edu/bearingdatacenter/pages/download-data-file). The data is sampled from 3 accelerometers (Drive End, Fan End and Base) on an industrial equipment at 12k hz.  However Normal data is missing on Base, so only Drive End and Fan End will be used. For training of auto-encoder, the Normal data will be used with some shifts to create large number of training data. To evaluate the fault classification capability, 30 data files from 10 types of faults and 3 types of Motor Load(HP)will be used as described in paper [1].  The 10 types of faults are Normal, Ball(0.007, 0.014, 0.021-inch faults), Inner Race (same 3 types as above), and Outer Race (same 3 types as above). One sample for the model is the form of 2,048x 2 (Drive End and Fan End) array. There are 19,800 test samples and 750 test samples. The 10 types of faults are completely balanced across training and test dataset. Test samples does not have any shift (overlap).

## Prerequisites

- Create ./data and ./data/download before running functions in DataProcessVib.py
- ​

### Reference

[1]          Wei Zhang,Gaoliang Peng, Chuanhao Li, Yuanhang Chen and Zhujin Zhang, “[A New Deep Learning Model forFault Diagnosis with Good Anti-Noise and Domain Adaption Ability on RawVibration Signals](http://www.mdpi.com/1424-8220/17/2/425)”, MDPI Sensors, 2017.

[2]          Kishore K.Reddy, Soumalya Sarkar, Vivek Venugopalan, Michael Giering, “[AnomalyDetection and Fault Disambiguation in Large Flight Data: A Multi-modal DeepAuto-encoder Approach](http://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2016/phmc_16_026.pdf)”, Annual Conference of the Prognostics and HealthManagement Society 2016.

[3]          AlirezaMakhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Fray, “[Adversarial Autoencoders](https://arxiv.org/pdf/1511.05644.pdf)”, 2016