# -*- coding: utf-8 -*-


from six.moves import cPickle as pickle


"""
Opening pickled datasets
"""

pfile = r"./data/ABCD_Datasets.pickle"
with (open(pfile, "rb")) as openfile:
    while True:
        try:
            ABCD_Datasets = pickle.load(openfile)
        except EOFError:
            break

X_test_A = ABCD_Datasets["test_datasets"]["A"]
Y_test_A = ABCD_Datasets["test_labels"]["A"]

X_test_B = ABCD_Datasets["test_datasets"]["B"]
Y_test_B = ABCD_Datasets["test_labels"]["B"]

X_test_C = ABCD_Datasets["test_datasets"]["C"]
Y_test_C = ABCD_Datasets["test_labels"]["C"]

X_train_C = ABCD_Datasets["train_datasets"]["C"]
Y_train_C = ABCD_Datasets["train_labels"]["C"]


"""
visual check
"""      
import matplotlib.pyplot as plt
import numpy as np
import random
"""
Showing n sampels for fault type i
pass dataset D for mixed motor Load
pass dataset A,B,C for motor load 1,2,3
datapoints default is 2048
"""

def ShowSamples(X_DS,Y_DS,Channel = None,datapoints = None):
    plt.rc('axes', grid=True)
    plt.rc('figure', figsize=(30, 3))
    plt.rc('legend', fancybox=True, framealpha=1)

    if datapoints is None:
        datapoints = 2048
    if Channel is None:
        Channel = 0
    for faulttype in range(1,11):
        idxTF = np.in1d(Y_DS,faulttype)
        idxNo = random.sample(list(np.where(idxTF)[0]),1)
        plt.subplot(1,10,faulttype)
        plt.plot(X_DS[idxNo[0]][:datapoints,Channel])
        plt.title("Fault Type "+str(faulttype))
    plt.show()
        
#ShowSamples(X_train_D,datapoints = 128)


#Channels = ['DE','FE'] 
c = 0

ShowSamples(X_DS = X_test_A, Y_DS = Y_test_A, Channel = c)
ShowSamples(X_DS = X_test_B, Y_DS = Y_test_B, Channel = c)
ShowSamples(X_DS = X_test_C, Y_DS = Y_test_C, Channel = c)

ShowSamples(X_DS = X_test_A, Y_DS = Y_test_A, Channel = c, datapoints=100)
ShowSamples(X_DS = X_test_B, Y_DS = Y_test_B, Channel = c, datapoints=100)
ShowSamples(X_DS = X_test_C, Y_DS = Y_test_C, Channel = c, datapoints=100)
