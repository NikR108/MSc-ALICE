
# This script performs hyperparameter optimization with a custom for-loop

#-----------------------------------------------
# Import the necessary packages 

import numpy as np
import matplotlib.pyplot as plt 
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

from keras.models import save_model
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



from sklearn.model_selection import train_test_split


X_train, X_valid, Y_train, Y_valid = train_test_split(sigs, labels, test_size=0.10, random_state=108, stratify=labels)

#********************************************
# save the training and testing splits as sep datafames

#training = pd.concat([X_train, y_train], axis=1)  # attach side by side 
#testing = pd.concat([X_test, y_test], axis=1)

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
	m.add(Dense(hn, activation=h_act, name="hidden_layer"))
	m.add(Dropout(drop2))
	m.add(Dense(2, activation="softmax", name='Dense-output'))


	# early stopping 
	#es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

	# Compile 
	#m.compile(loss="binary_crossentropy",
	#optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])


	return m


print("\nTraining model now ____------_____------_______----\n")
print("\n******************************************************\n")

cnn = create_model(nc1=120, ksize=3, act1="relu", mpsize=3, drop1=0.1, hn=64, h_act="relu", drop2=0.3)
						
cnn.summary()
# Compile 
cnn.compile(loss="binary_crossentropy",optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])
						
h = cnn.fit(x_train_scaled, y_train_cat, batch_size=256, epochs=50, verbose=1, validation_data=(x_valid_scaled, y_valid_cat), callbacks=es)

score = cnn.evaluate(x_valid_scaled, y_valid_cat, verbose=0)

print(score)

modelhist = pd.DataFrame(h.history)
modelhist.to_csv("cnn2_NET1_history.csv", index=False)

# save model
cnn.save("cnn2_NET1.h5")


	# Fit to data 
	#h = m.fit(x_train_scaled, Y_train, batch_size=128, epochs=50, 
	#verbose=1, validation_split=0.2, callbacks=es)


	#score = m.evaluate(x_test_scaled, Y_test, verbose=0)

	# append score[0] to loss np array
	# append score[1] to acc np array 
	# append score[2] to auc np array 





#try_nc1 = [10, 16, 20, 24, 30, 32, 34, 36, 38 ,40, 44, 50, 54, 60, 64, 70, 80, 90, 100, 110, 120]
#try_act1 = ["relu", "sigmoid", "tanh"]
#kernel = [2, 3]
#pool = [2,3]
#dr1 = [0.1, 0.2, 0.25, 0.3]

#dense1 = [None, 2, 4, 6, 8, 10, 16, 20, 30, 40, 50, 60, 70, 80]
#try_act2 =['relu', 'sigmoid', 'tanh']




# Now start the nested loops for hyperparameter optimization 
'''
def tunemodel():

# empty lists to store correspnding param
	KERNEL = []
	POOL = []
	DRATE = []
	ACT = []
	FILTER = []

	ACC = []
	LOSS = []
	AUC = []

	for k in kernel:
		for p in pool:
			for d in dr1:
				for act in try_act1:
					for filters in try_nc1:

						cnn = create_model(nc1=filters, ksize=k, act1=act, mpsize=p, drop1=d)
						es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

						# Compile 
						cnn.compile(loss="binary_crossentropy",
						optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])
						
						#h = cnn.fit(x_train_scaled, Y_train, batch_size=128, epochs=50, 
							#verbose=1, validation_split=0.1, callbacks=es)

						score = cnn.evaluate(x_test_scaled, Y_test, verbose=0)

						# record our model params 
						KERNEL.append(k)
						POOL.append(p)
						DRATE.append(d)
						ACT.append(act)
						FILTER.append(filters)


						# record our metrics 
						LOSS.append(h.history['val_loss'][-1])
						ACC.append(h.history['val_acc'][-1])
						AUC.append(h.history['val_auc'][-1])

	# Now convert our arrays to a nice dataframe 
	tune_results = pd.DataFrame({"kernel_size":KERNEL,
		"poolsize":POOL,
		"Dropout":DRATE,
		"Activation":ACT,
		"Filters":FILTER})

	tune_results.to_csv("tune1.csv", index=False)


	
'''





	

					






























