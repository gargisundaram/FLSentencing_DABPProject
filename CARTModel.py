#takes dataframe directly
#target is the name of the y column for the dataset 
#model is the type of sklearn model being run (ex: DecisionTreeClassifier)
#should change the random state but it's currently set to a default

def cart_model(df, target, model, seed):
  #split into x and y variables
  x = df.drop([target], axis = 1)
  y = df[[target]]

  #split into train and test set
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = seed)

  #get best params
  tree = model(random_state = seed)
  gs = GridSearchCV(model(), {'max_depth':[3,40], 'min_samples_split':[10,80]})
  grid_result = gs.fit(x_train, y_train.values.ravel())
  best_params = grid_result.best_params_
  
  #run model
  tree = model(max_depth = best_params.get('max_depth'), max_features = best_params.get('max_features'), random_state=seed)
  t = tree.fit(x_train, y_train.values.ravel())
  y_pred = t.predict(x_test)
  tree_train = tree.score(x_train, y_train)
  tree_test = tree.score(x_test, y_test)

  #results
  if model == DecisionTreeClassifier:
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
  elif model == DecisionTreeRegressor:
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  else:
    print("This model only works for CART models! Pick one of them and run again")

  #feature importance
  importance = pd.DataFrame(tree.feature_importances_, columns = ["Importance"])
  importance['Features'] = x.columns
  importance = importance[importance['Importance'] > 0].sort_values(by = 'Importance', ascending = False)

  # Print table of feature importances
  display(importance)

  # Histogram of feature importances
  plt.bar(importance['Features'], importance['Importance'])
  plt.xticks(rotation = -90)
  plt.show()

  #Tree plot
  plt.figure(figsize=(25,15))
  plot_tree(tree, feature_names = x.columns, filled = True)
  
  return best_params, t