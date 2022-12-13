import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import shap

#takes dataframe directly
#target is the name of the y column for the dataset 
#model is the type of sklearn model being run (ex: DecisionTreeClassifier)
#should change the random state but it's currently set to a default

def get_tree(df, target, model, paramdict, seed = 10, nsample = None):
  if nsample.isna() == False:
    df = df.sample(n = nsample)
  
  #split into x and y variables
  x = df.drop([target], axis = 1)
  y = df[[target]]

  #split into train and test set
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = seed)
  y_train = y_train.values.ravel()
  y_test = y_test.values.ravel()

  #get best params
  tree = model(random_state = seed)
  gs = GridSearchCV(model(), paramdict)
  grid_result = gs.fit(x_train, y_train)
  best_params = grid_result.best_params_
  
  #run model
  tree = model(**best_params, random_state=seed)
  t = tree.fit(x_train, y_train)
  y_pred_train = t.predict(x_train)
  y_pred_test = t.predict(x_test)


  #results
  if (model == DecisionTreeClassifier) or (model == RandomForestClassifier) or (model == XGBClassifier):
    metric = {'conf_mat':confusion_matrix(y_test, y_pred_test),
              'class_report':classification_report(y_test, y_pred_test),
              "accuracy" : accuracy_score(y_test, y_pred_test)}
    print(
          "Confusion matrix:\n", metric['conf_mat'],
          "Classification Report:\n", metric['class_report'],
          "Accuracy:", metric['accuracy'])

  elif (model == DecisionTreeRegressor) or (model == RandomForestRegressor) or (model == XGBRegressor):
    metric = {'train_rmse' : np.sqrt(mean_squared_error(y_train, y_pred_train)), 
              'test_rmse' : np.sqrt(mean_squared_error(y_test, y_pred_test))}
    print('train RMSE:', metric['train_rmse'], '\ntest RMSE:', metric['test_rmse'])

  #feature importance
  importance = pd.DataFrame(tree.feature_importances_, columns = ["Importance"])
  importance['Features'] = x.columns
  importance = importance[importance['Importance'] > 0].sort_values(by = 'Importance', ascending = False)

  # Print table of feature importances
  print("Feature Importance Table")
  display(importance)

  # Histogram of feature importances
  plt.bar(importance['Features'], importance['Importance'])
  plt.xticks(rotation = -90)
  print("Histogram of Feature Importance")
  plt.show()

  #Beeswarm plots with shap values
  explainer = shap.TreeExplainer(t)
  shap_values = explainer.shap_values(x_test)
  print("Feature Beeswarm Plot")
  shap.summary_plot(shap_values, x_test)

  return t, metric, importance, shap_values
  