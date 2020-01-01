""" this file contains the code for a randomForestClassifier """
from sklearn.ensemble import RandomForestClassifier
from preprocess_data import return_when_called
from sklearn.metrics import mean_absolute_error, classification_report

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()

#creating instance of the classifier
clf = RandomForestClassifier(n_estimators = 100)

#fitting the model
clf.fit(X_train_scaled,y_train)

#making predictions on validation set
val_predictions = clf.predict(X_val_scaled)

#calculating mean absolute error
mae = mean_absolute_error(y_val,val_predictions)
print("mean absolute error is: " +str(mae))

#making predictions on test set and printing classification report
test_predictions = clf.predict(X_test_scaled)
print(classification_report(y_test,test_predictions))