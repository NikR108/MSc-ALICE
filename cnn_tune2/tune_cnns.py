
# This script performs hyperparameter optimization with a custom for-loop

#-----------------------------------------------
# Import the necessary packages 

import numpy as np
 
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense 
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
from keras import layers
#from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as K
from keras.utils import np_utils

#-------------------------------------------------





# by default 0 is negative and 1 is the positive class
# We're searching for the sim data so we consider it as the positive class 
#.................................................


# Load the data 

TRD = pd.read_csv("../ML_prep_data/train.csv")

sigs = TRD.iloc[:, :90]
labels = TRD.label

#X_train = sigs
#Y_train = labels 


import sklearn 
from sklearn.model_selection import train_test_split


X_train, X_valid, Y_train, Y_valid = train_test_split(sigs, labels, test_size=0.10, random_state=108, stratify=labels)

#********************************************
# save the training and validation splits as sep datafames

actual_train = pd.concat([X_train, Y_train], axis=1)  # attach side by side 
validation = pd.concat([X_valid, Y_valid], axis=1)

actual_train.to_csv("actual_train.csv", index=False)
validation.to_csv("validation.csv", index=False)

#********************************************




# rehape the signals into 3x30 arrays 
x_train = np.array(X_train).reshape(len(X_train), 3, 30)
x_valid = np.array(X_valid).reshape(len(X_valid), 3, 30)


# Normalize the signal arrays 
x_train_scaled = x_train/1023
x_valid_scaled = x_valid/1023


# One hot encode the labels 
y_train = np.array(Y_train)
y_valid = np.array(Y_valid)

y_train_cat = keras.utils.np_utils.to_categorical(y_train, 2)
y_valid_cat = keras.utils.np_utils.to_categorical(y_valid, 2)








#=================================================================
# define a function to instantiate a keras model 

es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

def create_model(nc1, ksize ,act1, mpsize,  drop1, hn, h_act, drop2):
	m = Sequential()
	m.add(Conv2D(nc1, kernel_size=(ksize,ksize), padding='same',activation=act1, input_shape=(3,30,1), name='convol2D'))
	m.add(MaxPooling2D(pool_size=(mpsize,mpsize), name="maxpool"))
	m.add(Flatten())
	m.add(Dropout(drop1))
	m.add(Dense(hn, activation=h_act, name="hidden-layer"))
	m.add(Dropout(drop2))
	m.add(Dense(2, activation="softmax", name='Dense-output'))


	# early stopping 
	#es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

	# Compile 
	m.compile(loss="binary_crossentropy",
	optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])


	return m

	# Fit to data 
	#h = m.fit(x_train_scaled, Y_train, batch_size=128, epochs=50, 
	#verbose=1, validation_split=0.2, callbacks=es)


	#score = m.evaluate(x_test_scaled, Y_test, verbose=0)

	# append score[0] to loss np array
	# append score[1] to acc np array 
	# append score[2] to auc np array 



try_nc1 = [64, 80, 120]


try_act1 = ["relu"]
kernel = [3]
pool = [3]
dr1 = [0.1]
dr2 = [0.1, 0.2, 0.3]
dense1 = [6, 12, 24, 32, 64]
try_act2 =['relu', 'sigmoid', 'tanh']




# Now start the nested loops for hyperparameter optimization 

def tunemodel():

# empty lists to store correspnding param

	KERNEL = []
	POOL = []
	DRATE1 = []
	DRATE2 = []
	ACT1 = []
	ACT2 = []
	FILTER = []
	HNEURONS = []

	ACC = []
	LOSS = []
	AUC = []

	for k in kernel:
		for p in pool:
			for d in dr1:
				for act in try_act1:
					for filters in try_nc1:
						for d2 in dr2:
							for act2 in try_act2:
								for node in dense1:

									# print the params beig used
									print("\n--------------------------")
									print("kernel = {}\npoolsize = {}\ndropout = {}\nactivation = {}\nn_filters = {}\ndropout2 = {}\nhidden_act = {}\nhidden_neurons={}".format(k,p,d,act,filters,d2,act2, node)) 
									print("---------------------------\n")	

									cnn = create_model(nc1=filters, ksize=k, act1=act, mpsize=p, drop1=d, drop2=d2, h_act=act2, hn=node)
									es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

									# Compile 
									#cnn.compile(loss="binary_crossentropy",
									#optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])
						
									h = cnn.fit(x_train_scaled, y_train_cat, batch_size=256, epochs=10, verbose=1, validation_data=(x_valid_scaled, y_valid_cat), callbacks=es)

									score = cnn.evaluate(x_valid_scaled, y_valid_cat, verbose=0)

									# record our model params
									print("\n---Recording model params----")
									print("-------------------------------") 
									KERNEL.append(k)
									POOL.append(p)
									DRATE1.append(d)
									DRATE2.append(d2)
									HNEURONS.append(node)
									ACT1.append(act)
									ACT2.append(act2)
									FILTER.append(filters)
						
									print("\n---Recording validation metrics---")
									print("----------------------------------")
									# record our metrics 
									LOSS.append(score[0])
									ACC.append(score[1])
									AUC.append(score[2])

	# Now convert our arrays to a nice dataframe 
	tune_results = pd.DataFrame({"kernel_size":KERNEL,
		"poolsize":POOL,
		"Dropout_1":DRATE1,
		"Activation_conv2D":ACT1,
		"Filters":FILTER,
		"Dropout_2":DRATE2,
		"Hidden_neurons":HNEURONS,
		"Dense_Activation":ACT2,
		"val_loss":LOSS,
		"val_acc":ACC,
		"val_auc_pr":AUC})

	tune_results.to_csv("tune2.csv", index=False)


	
tunemodel()





	

					






























