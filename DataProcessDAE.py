# -*- coding: utf-8 -*-

"""
Output is ./data/DAE_NASA.pickle, 1.35G file
Prerequisite: Create ./data and ./data/download
"""
import os
from six.moves.urllib.request import urlretrieve

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

"""
Data download
Skippes if files are already downloaded.
http://csegroups.case.edu/bearingdatacenter/pages/download-data-file
"""

url_base =  'http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/'
download_folder = r'./data/download'
# category lable 1 to 10
# Data Set A, B, C (D will be created later with 2nd half of A,B,C)
file_dictionary = {
        "1-A" : "98.mat",
        "1-B" : "99.mat",
        "1-C" : "100.mat",
        "2-A" : "119.mat",
        "2-B" : "120.mat",
        "2-C" : "121.mat",
        "3-A" : "186.mat",
        "3-B" : "187.mat",
        "3-C" : "188.mat",
        "4-A" : "223.mat",
        "4-B" : "224.mat",
        "4-C" : "225.mat",
        "5-A" : "106.mat",
        "5-B" : "107.mat",
        "5-C" : "108.mat",
        "6-A" : "170.mat",
        "6-B" : "171.mat",
        "6-C" : "172.mat",
        "7-A" : "210.mat",
        "7-B" : "211.mat",
        "7-C" : "212.mat",
        "8-A" : "131.mat",
        "8-B" : "132.mat",
        "8-C" : "133.mat",
        "9-A" : "198.mat",
        "9-B" : "199.mat",
        "9-C" : "200.mat",
        "10-A" : "235.mat",
        "10-B" : "236.mat",
        "10-C" : "237.mat"
        }

for key,filename in file_dictionary.items():
    dest_filename = os.path.join(download_folder, filename)
    if not os.path.exists(dest_filename):
        print('\nAttempting to download:', filename) 
        filename, _ = urlretrieve(url_base + filename, dest_filename)
        print('\nDownload Complete!')
        statinfo = os.stat(dest_filename)
        print(filename,':', statinfo.st_size)

"""
getting arrays
Assuming they only used DE (drive end accelerometer) for bearing fault detection
-> get 2 channels. No BA data on Normal(1-) 
"""

Channels = ['DE','FE'] 
Vib_mc = {} # multi-channel

for c in Channels:
    Sen_series = {}
    for key,filename in sorted(file_dictionary.items()):
        mtfilename = os.path.join(download_folder, filename)
        data = scipy.io.loadmat(mtfilename)
        for i in data:
            if  '_'+c+'_time' in i:
                Sen_series[key] = data[i]
    Vib_mc[c]= Sen_series

#np.savetxt('test.csv',Vib_series['FE']['2-A'],delimiter=',')

"""
checking # of rows, mean, min, max
minimum row
"""       
Channel = Channels[1]

minrow = Vib_mc[Channel][list(Vib_mc[Channel].keys())[0]].shape[0]

for key,array in sorted(Vib_mc[Channel].items()):
    print(key, "rows:", array.shape[0]
        , "avg:",round(np.mean(array),2)
        , "min:",round(np.min(array),2)
        , "max:",round(np.max(array),2))    
    if array.shape[0] < minrow:
        minrow = array.shape[0]

print("minimum row:",minrow )

"""
visual check
"""      
sample = Vib_mc['FE']['2-A']
plt.plot(sample[:128])
plt.show()
sample = Vib_mc['DE']['2-A']
plt.plot(sample[:128])
plt.show()
#plt.plot(sample[:524])
#plt.show()

"""
Creating A,B,C,D dataset
Refer to the paper, 4.1 Data Description
Based on minimum row, shift we can use is up to 103. Let's use 103.
Assuming the condition did not change during sampling, 
we take test samples without shift(augument) from the 1st half.
"""
# take 25 samples ( each 2048 data points) from each series
# to fit the keras input, numpy array shape to be (batch_size=25,steps=2048,input_dim = 3)

X_test_temp = {}
sample_len = 2048
sample_test_cnt = 25

for key,sample in sorted(Vib_mc[Channels[0]].items()):
    X_test_temp[key] = np.concatenate((Vib_mc[Channels[0]][key][:sample_len*sample_test_cnt].reshape(sample_test_cnt,sample_len,1)
                            ,Vib_mc[Channels[1]][key][:sample_len*sample_test_cnt].reshape(sample_test_cnt,sample_len,1)
                            ),axis=2)
    # may need to enhance to dynamically understand the number of channels
    
# augumenting train data
X_train_temp = {}
sample_train_cnt = 660
sample_train_shift = 103
for key,sample in sorted(Vib_mc[Channels[0]].items()):
    train_array = np.concatenate((Vib_mc[Channels[0]][key][sample_len*sample_test_cnt:sample_len*sample_test_cnt+sample_len].reshape(1,sample_len,1)
                                 ,Vib_mc[Channels[1]][key][sample_len*sample_test_cnt:sample_len*sample_test_cnt+sample_len].reshape(1,sample_len,1)
                                 ),axis=2)
    for i in range(sample_train_cnt-1):
        train_array = np.concatenate((train_array
                                 ,np.concatenate((
                                         Vib_mc[Channels[0]][key][sample_len*sample_test_cnt+sample_train_shift*(i+1):sample_len*sample_test_cnt+sample_len+sample_train_shift*(i+1)].reshape(1,sample_len,1)
                                        ,Vib_mc[Channels[1]][key][sample_len*sample_test_cnt+sample_train_shift*(i+1):sample_len*sample_test_cnt+sample_len+sample_train_shift*(i+1)].reshape(1,sample_len,1)
                                ),axis=2)),axis=0)
    X_train_temp[key] = train_array

# Creating A, B, C dataset with lable. one-hot encoding later when used

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def create_dataset(X_temp):
    X = {"A":np.empty((0,sample_len,2),float),"B":np.empty((0,sample_len,2),float),"C":np.empty((0,sample_len,2),float)}
    Y = {"A":np.empty((0,1),int),"B":np.empty((0,1),int),"C":np.empty((0,1),int)}
    for key,sample in sorted(X_temp.items()):
        X[key.split(sep="-")[1]]=np.concatenate((X[key.split(sep="-")[1]],sample),axis=0)
        y = np.empty(len(sample),int)
        y.fill(key.split(sep="-")[0])
        y = y.reshape((len(sample),1))
        Y[key.split(sep="-")[1]]=np.concatenate((Y[key.split(sep="-")[1]],y),axis=0)
    #shuffling
    for key,samplearray in sorted(X.items()):
        X[key],Y[key] = randomize(samplearray,Y[key])
    return X,Y

X_test,Y_test = create_dataset(X_test_temp) 
X_train,Y_train = create_dataset(X_train_temp)

# Creating D dataset
X_test["D"],Y_test["D"] = randomize(np.vstack((X_test["A"],X_test["B"],X_test["C"]))
    ,np.vstack((Y_test["A"],Y_test["B"],Y_test["C"])))

X_train["D"],Y_train["D"] = randomize(np.vstack((X_train["A"],X_train["B"],X_train["C"]))
    ,np.vstack((Y_train["A"],Y_train["B"],Y_train["C"])))

"""
Saving the dataset as pickle format
"""

pickle_file = os.path.join('./data/', 'ABCD_Datasets.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_datasets': X_train,
    'train_labels': Y_train,
    'test_datasets': X_test,
    'test_labels': Y_test,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
