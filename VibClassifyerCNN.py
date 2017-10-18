# -*- coding: utf-8 -*-

"""
Input ABCD_Datasets.pickle from DataProcess.py
Output model.weights.best.hdf5

"""
from six.moves import cPickle as pickle
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
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

"""
We are using dataset D.

"""
X_train_D = ABCD_Datasets["train_datasets"]["D"]
Y_train_D = ABCD_Datasets["train_labels"]["D"]
X_test_D = ABCD_Datasets["test_datasets"]["D"]
Y_test_D = ABCD_Datasets["test_labels"]["D"]

"""
one hot-encoding
validation dataset
"""
# one-hot encode the labels
num_classes = 10
Y_train_D_hot = keras.utils.to_categorical(Y_train_D-1, num_classes)
Y_test_D_hot = keras.utils.to_categorical(Y_test_D-1, num_classes)

# break training set into training and validation sets
(X_train, X_valid) = X_train_D[2000:], X_train_D[:2000]
(Y_train, Y_valid) = Y_train_D_hot[2000:], Y_train_D_hot[:2000]
X_test = X_test_D
Y_test = Y_test_D_hot

# print shape of training set
print('x_train shape:', X_train.shape)

# print number of training, validation, and test images
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')

"""
CNN modeling
1 Channel
"""
CNNch = 1

model = Sequential()
#1
model.add(Conv1D(filters=16, kernel_size=64,strides = 16, padding='same', activation='relu', 
                        input_shape=(2048, CNNch)))
model.add(MaxPooling1D(pool_size=2))
#2
model.add(Conv1D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#3
model.add(Conv1D(filters=64, kernel_size=3, strides = 1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
#4
model.add(Conv1D(filters=64, kernel_size=3, strides = 1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
#5
model.add(Conv1D(filters=64, kernel_size=3, strides = 1, padding='same', activation='relu'))
#paper no padding?, Yes, to make 5th layer output 6 width and 3 after pooling
#-> same seems to perform little better because of more parameter? 
# little diffrernt from the paper but keep it as padding = 'same'
model.add(MaxPooling1D(pool_size=2))  

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

"""
CNN training
20 epochs, test/train/validation accuracy 100%
without learning, 10%.
"""
# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = model.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=1,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

"""
Evaluating accuracty on test
"""
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# evaluate and print test accuracy
score = model.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'CNN Test accuracy:', score[1])

score = model.evaluate(X_train[:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'CNN train accuracy:', score[1])

score = model.evaluate(X_valid[:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'CNN validation accuracy:', score[1])

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNN modeling
1 Channel, model adjusted
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNNch = 1

modelC = Sequential()
#1
modelC.add(Conv1D(filters=64, kernel_size=32,strides = 8, padding='same', activation='relu', 
                        input_shape=(2048, CNNch)))
modelC.add(MaxPooling1D(pool_size=2))
modelC.add(Dropout(0.2))
#2
modelC.add(Conv1D(filters=16, kernel_size=4, strides = 1, padding='same', activation='relu'))
modelC.add(MaxPooling1D(pool_size=2))
modelC.add(Dropout(0.2))
#3

modelC.add(Conv1D(filters=8, kernel_size=2, strides = 1, padding='same', activation='relu'))
modelC.add(MaxPooling1D(pool_size=2))
modelC.add(Dropout(0.3))

modelC.add(Flatten())
modelC.add(Dense(100, activation='relu'))
modelC.add(Dropout(0.2))
modelC.add(Dense(10, activation='softmax'))

modelC.summary()


# compile the model
modelC.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])


# train the model
checkpointer = ModelCheckpoint(filepath='modelC.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = modelC.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=20,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

# load the weights that yielded the best validation accuracy
modelC.load_weights('modelC.weights.best.hdf5')

# evaluate and print test accuracy
score = modelC.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'CNN Test accuracy:', score[1])

score = modelC.evaluate(X_train[:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'CNN train accuracy:', score[1])

score = modelC.evaluate(X_valid[:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'CNN validation accuracy:', score[1])


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNN modeling
2 Channels
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CNNch = 2

modelC2 = Sequential()
#1
modelC2.add(Conv1D(filters=16*16, kernel_size=64,strides = 16, padding='same', activation='relu', 
                        input_shape=(2048, CNNch)))
modelC2.add(MaxPooling1D(pool_size=2))
#2
modelC2.add(Conv1D(filters=16, kernel_size=3, strides = 1, padding='same', activation='relu'))
modelC2.add(MaxPooling1D(pool_size=2))
#3
modelC2.add(Conv1D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu'))
modelC2.add(MaxPooling1D(pool_size=2))
modelC2.add(Dropout(0.2))
#4
modelC2.add(Conv1D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu'))
modelC2.add(MaxPooling1D(pool_size=2))
modelC2.add(Dropout(0.2))
#5
modelC2.add(Conv1D(filters=32, kernel_size=3, strides = 1, padding='same', activation='relu'))
#paper no padding?, Yes, to make 5th layer output 6 width and 3 after pooling
#-> same seems to perform little better because of more parameter? 
# little diffrernt from the paper but keep it as padding = 'same'
modelC2.add(MaxPooling1D(pool_size=2))  

modelC2.add(Flatten())
modelC2.add(Dense(50, activation='relu'))
modelC2.add(Dropout(0.2))
modelC2.add(Dense(10, activation='softmax'))

modelC2.summary()


# compile the model
modelC2.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

# train the model
checkpointer = ModelCheckpoint(filepath='CNNC2.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = modelC2.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=8,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

# load the weights that yielded the best validation accuracy
modelC2.load_weights('CNNC2.weights.best.hdf5')

# evaluate and print test accuracy
score = modelC2.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'CNN Test accuracy:', score[1])

score = modelC2.evaluate(X_train[:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'CNN train accuracy:', score[1])

score = modelC2.evaluate(X_valid[:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'CNN validation accuracy:', score[1])

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Ref
Logistic regression
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# define the model
modelLog = Sequential()
modelLog.add(Flatten(input_shape = (2048,CNNch)))
modelLog.add(Dense(10, activation='softmax'))

modelLog.summary()
# total param 2,566,642
# ref  CNN       53,830
modelLog.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='Log.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = modelLog.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=1,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

modelLog.load_weights('Log.weights.best.hdf5')

score = modelLog.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'Logistic Regression Test accuracy:', score[1])


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
1 hiden layer
try different structure as you wish
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# define the model
modelH1MLP = Sequential()
modelH1MLP.add(Flatten(input_shape = (2048,CNNch)))
modelH1MLP.add(Dense(3000, activation='relu'))
modelH1MLP.add(Dropout(0.2))
modelH1MLP.add(Dense(10, activation='softmax'))

modelH1MLP.summary()

modelH1MLP.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='H1MLP.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = modelH1MLP.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=1,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

modelH1MLP.load_weights('H1MLP.weights.best.hdf5')

score = modelH1MLP.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

score = modelH1MLP.evaluate(X_train[:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'train accuracy:', score[1])

score = modelH1MLP.evaluate(X_valid[:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'validation accuracy:', score[1])


"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
2 hiden layers
try different structure as you wish
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# define the model
modelH2MLP = Sequential()
modelH2MLP.add(Flatten(input_shape = (2048,CNNch)))
modelH2MLP.add(Dense(260, activation='relu'))
modelH2MLP.add(Dropout(0.2))
modelH2MLP.add(Dense(260, activation='relu'))
#modelH2MLP.add(Dropout(0.2))
modelH2MLP.add(Dense(10, activation='softmax'))

modelH2MLP.summary()
# total param 2,566,642
# ref  CNN       60,230
modelH2MLP.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='H2MLP.weights.best.hdf5', verbose=1, 
                               save_best_only=True)

hist = modelH2MLP.fit(X_train[:,:,0:CNNch], Y_train, batch_size=32, epochs=1,
          validation_data=(X_valid[:,:,0:CNNch], Y_valid), callbacks=[checkpointer], 
          verbose=1, shuffle=True)

modelH2MLP.load_weights('H2MLP.weights.best.hdf5')

score = modelH2MLP.evaluate(X_test[:,:,0:CNNch], Y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

score = modelH2MLP.evaluate(X_train[:,:,0:CNNch], Y_train, verbose=0)
print('\n', 'train accuracy:', score[1])

score = modelH2MLP.evaluate(X_valid[:,:,0:CNNch], Y_valid, verbose=0)
print('\n', 'validation accuracy:', score[1])

