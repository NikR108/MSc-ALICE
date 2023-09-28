
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

# import util to save model 
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




# rehape the signals into 3x30 arrays <--------- No need we training a ANN 
x_train = np.array(X_train)
x_valid = np.array(X_valid)


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

def create_model(hn, act, drop1):
	m = Sequential()
	m.add(Dense(hn, input_shape=(90,), activation=act, name="hidden1"))
	m.add(Dropout(drop1))
	m.add(Dense(2, activation="softmax", name='Dense-output'))


	# early stopping 
	#es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

	# Compile 
	m.compile(loss="binary_crossentropy",
	optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])


	return m



# create an instance of a keras ann model 
ann = create_model(hn=120, act="relu", drop1=0.1)

# compile model 
# this is already done by our function 

# fit the model 
h = ann.fit(x_train_scaled, y_train_cat, batch_size=256, epochs=50,verbose=1, validation_data=(x_valid_scaled, y_valid_cat), callbacks=es)

# evaluate the model on the validation data 
score = ann.evaluate(x_valid_scaled, y_valid_cat, verbose=0)
print(score)



# record the model training history
model_hist = pd.DataFrame(h.history)

model_hist.to_csv("ann1_history.csv", index=False)


# save the model 

ann.save("ann1.h5")



	# Fit to data 
	#h = m.fit(x_train_scaled, Y_train, batch_size=128, epochs=50, 
	#verbose=1, validation_split=0.2, callbacks=es)


	#score = m.evaluate(x_test_scaled, Y_test, verbose=0)

	# append score[0] to loss np array
	# append score[1] to acc np array 
	# append score[2] to auc np array 



#h1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
#fn = ['relu', 'tanh', 'sigmoid']
#dr = [0.1, 0.2, 0.3]




# Now start the nested loops for hyperparameter optimization 
'''
def tunemodel():

# empty lists to store correspnding param

	NODES = []
	DRATE = []
	ACT = []
	

	ACC = []
	LOSS = []
	AUC = []

	for d in dr:
		for f in fn:
			for n in h1:
		

				# print the params beig used
				print("\n--------------------------")
				print("Neurons={}\nDropout={}\nAct={}".format(n,d,f))		 
				print("---------------------------\n")	

				ann = create_model(hn=n, act=f, drop1=d)
				es = EarlyStopping(monitor = 'val_loss', mode='auto', patience=5)

						
				h = ann.fit(x_train_scaled, y_train_cat, batch_size=256, epochs=12, verbose=1, validation_data=(x_valid_scaled, y_valid_cat), callbacks=es)

				score = ann.evaluate(x_valid_scaled, y_valid_cat, verbose=0)
				print(score)

				# record our model params
				print("\n---Recording model params----")
				print("-------------------------------") 
				NODES.append(n)
				DRATE.append(d)
				ACT.append(f)
			
						
				print("\n---Recording validation metrics---")
				print("----------------------------------")
				# record our metrics 
				LOSS.append(score[0])
				ACC.append(score[1])
				AUC.append(score[2])

	# Now convert our arrays to a nice dataframe 
	tune_results = pd.DataFrame({"Hidden_neurons":NODES,
		"Dropout":DRATE,
		"Activation":ACT,
		"val_loss":LOSS,
		"val_acc":ACC,
		"val_auc_pr":AUC})

	tune_results.to_csv("ANNtune1.csv", index=False)


	
tunemodel()

'''



	

					






























