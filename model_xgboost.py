""" this file has the code for a classifier using xgboostclassifier """
from xgboost import XGBClassifier
from preprocess_data import return_when_called
from sklearn.metrics import mean_absolute_error,classification_report

#gets the X and Y arrays from preprocess_data
X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val = return_when_called()

#instance of the classifier
xgb = XGBClassifier(n_estimators = 3000, learning_rate = 0.01)

#fitting the model
xgb.fit(X_train_scaled,y_train)

#making predictions on validation set
val_predictions = xgb.predict(X_val_scaled)

#calculating mean absolute error on validation set and printing it
mae = mean_absolute_error(val_predictions,y_val)
print("the mean absolute error is: " + str(mae))

#making predictions on the test set
test_predictions = xgb.predict(X_test_scaled)

#printing classification report
print(classification_report(y_test,test_predictions))