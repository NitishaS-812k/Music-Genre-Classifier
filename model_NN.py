""" this file defines the neural network model """

#required imports
from keras import models
from keras import layers
from preprocess_data import return_when_called
import numpy as np
from keras.utils import to_categorical

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

model = models.Sequential()
model.add(layers.Dense(26,activation = 'relu')) #input layer
model.add(layers.Dense(20,activation = 'relu')) #hidden layer
model.add(layers.Dense(14,activation = 'softmax')) #output layer

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(np.array(X_train_scaled),
                    np.array(y_train),
                    epochs=50,
                    batch_size=100,
                    validation_data=(np.array(X_val_scaled),np.array(y_val)))

#evaluating model on test set                   
test_loss, test_acc = model.evaluate(np.array(X_test_scaled),np.array(y_test)) 
print('test_acc: ',test_acc)