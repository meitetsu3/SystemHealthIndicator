# -*- coding: utf-8 -*-
"""
Downloading and creating training data descirbed in 
http://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2016/phmc_16_026.pdf
Output is ./data/DAE_NASA_train.pickle, 10M file
Prerequisite: Create ./data and ./data/download
"""

import os
from six.moves.urllib.request import urlretrieve
import numpy as np
from six.moves import cPickle as pickle
import zipfile
import glob
import pandas as pd
import collections

"""
Data download and unzip
Skippes if files are already downloaded.
"""
url_base =  'https://c3.nasa.gov/dashlink/static/media/dataset/'
download_folder = r'./data/download/'
filename = '2010_09_01.zip'
dest_filename = os.path.join(download_folder, filename)

if not os.path.exists(dest_filename):
    print('\nAttempting to download:', filename) 
    filename, _ = urlretrieve(url_base + filename, dest_filename)
    print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    print(filename,':', statinfo.st_size)
    zip_ref = zipfile.ZipFile(dest_filename, 'r')
    zip_ref.extractall(download_folder)
    zip_ref.close()

"""
gathering data to a dataframe
Paper forgot to mention Motor X Temperature in the text, but appears on graphs
13 variabls + Time
"""
# column name and index
ColLables = collections.OrderedDict()
ColLables['Time'] = 0
ColLables['Actuator Z Position'] = 4
ColLables['Measured Load'] = 6
ColLables['Motor X Current'] = 7
ColLables['Motor Y Current'] = 8
ColLables['Motor Z Current'] = 9
ColLables['Motor X Voltage'] = 10
ColLables['Motor Y Voltage'] = 11
ColLables['Motor X Temperature'] = 13 # paper forgot to list this
ColLables['Motor Y Temperature'] = 14
ColLables['Motor Z Temperature'] = 15
ColLables['Nut X Temperature'] = 16
ColLables['Nut Y Temperature'] = 17
ColLables['Ambient Temperature'] = 19

train_df = pd.DataFrame(columns=ColLables.keys())

for filename in glob.glob(os.path.join(download_folder,'2010_09_01/sdata/*_Nominal_Low.data')):
    with open(filename, 'r') as f:
        dft = pd.read_csv(f,sep='\t',header=None).loc[:,ColLables.values()]
        dft.columns = ColLables.keys()
        dft['Time'] = dft['Time'].map(lambda x: x.rstrip('_-0700'))
    train_df = train_df.append(dft,ignore_index=True)

# save to csv once. 95230 rows, 17M
# it is already sorted by Time
# df.to_csv('./data/train.csv')

"""
Shifging by 1 and create (:,50,13) array
Shuffle on 0 axis
"""
# exclude time
train_ar = train_df.loc[:, train_df.columns != 'Time'].as_matrix()
print(train_ar.shape) # (95230,13)
permutation = np.random.permutation(train_ar.shape[0])
train_ar = train_ar[permutation,:]

"""
Saving the dataset as pickle format
"""
pickle_file = os.path.join('./data/', 'DAE_NASA_train.pickle')
try:
  f = open(pickle_file, 'wb')
  save = train_ar
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
