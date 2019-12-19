""" this file defines the neural network model """
from keras import models
from keras import layers
from preprocess_data import return_when_called
import numpy as np

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()

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
test_loss, test_acc = model.evaluate(np.array(X_test_scaled),np.array(y_test)) #evaluating model on test set
print('test_acc: ',test_acc)