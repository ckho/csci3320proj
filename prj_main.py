from preprocess import transform
from preprocess import fill_missing

from lr import LogisticRegression
from naive_bayes import NaiveBayes

from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer

from MICE import MICEImputer

import numpy as np
import pandas as pd

def main():
  # load training data
  filename_train = './data/train.csv'
  train_dataset = transform(filename_train)
  X = train_dataset['data']
  y = train_dataset['target']

  X = X.drop('UserID', 1)

  X.loc[X.YOB < 1920, 'YOB'] = 0
  X.loc[X.YOB > 2004, 'YOB'] = 0
  X.loc[X.YOB.isnull(), 'YOB'] = 0
  X.loc[X['Income'].isnull(), 'Income'] = 0
  X.loc[X['HouseholdStatus'].isnull(), 'HouseholdStatus'] = 0
  X.loc[X['EducationLevel'].isnull(), 'EducationLevel'] = 0
  X.loc[X['Party'].isnull(), 'Party'] = 0

  X = X.apply(pd.to_numeric, errors='ignore')

  imput_model = MICEImputer(verbose=True, random_state=1, min_value=0, max_value=1, random_state=1)

  X_filled = imput_model.fit_transform(X)

  X_filled_df = pd.DataFrame(data=X_filled[:,:], columns=X.columns)

  # fill in missing data (optional)
  # X_full = fill_missing(X, 'most_frequent', False)

  numeric_cols = ['YOB', 'votes']
  X_num = X_filled_df[numeric_cols].as_matrix()

  # scale to <0,1>
  max_X = np.amax(X_num, 0)
  X_num = X_num / max_X

  cat_X = X_filled_df.drop(numeric_cols, axis = 1 )
  X_cat_X = cat_X.T.to_dict().values()

  # ### vectorize
  vectorizer = DictVectorizer(sparse = False)
  vec_X_cat = vectorizer.fit_transform(X_cat_X)


  X_combined = np.hstack((X_num, vec_X_cat))

  # ### cross-validation
  X_train, X_verify, y_train, y_verify = train_test_split(X_combined, y, test_size=0.2, random_state=0)

  # ### use the logistic regression
  print('Train the logistic regression classifier')
  lr_model = LogisticRegression()
  lr_model.fit(X_train, y_train)
  print(lr_model.score(X_verify, y_verify))


  # ### use the naive bayes
  print('Train the naive bayes classifier')
  nb_model = NaiveBayes()
  nb_model.fit(X_train, y_train)
  print(nb_model.score(X_verify, y_verify))

  # ## use the svm
  print('Train the SVM classifier')
  svm_model = svm.SVC(random_state=1, shrinking=True, probability=True)
  C_range = np.linspace(0.001, 100, 1000)
  gamma_range = np.linspace(0.0001, 1.000, 1000)
  param_grid = {'C': C_range, 'gamma': gamma_range}
  lrgs = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=6, n_jobs=-1)
  lrgs.fit(X_train, y_train)
  # svm_model.fit(X_train, y_train)
  print(lrgs.score(X_verify, y_verify))


  ## use the random forest
  print('Train the random forest classifier')
  rf_model = ensemble.RandomForestClassifier(random_state=1)
  rf_model.fit(X_train, y_train)
  print(rf_model.score(X_verify, y_verify))


  # ## get test data
  # filename_train = './data/test.csv'
  # test_dataset = transform_for_test(filename_test)
  # X_test = test_dataset['data']

  # # TO DO: Fill Missing

  # predictions_path = './predictions/'

  # ## do predictions
  # lr_predictions = lr_model.predict(X_test)
  # lr_predictions_file = open(predictions_path + 'lr_predictions.csv', 'w')
  # print('UserID,Happy', file=lr_predictions_file)
  # for x_test, prediction in zip(X_test, lr_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=lr_predictions_file)

  # nb_predictions = nb_model.predict(X_test)
  # nb_predictions_file = open(predictions_path + 'nb_predictions.csv', 'w')
  # print('UserID,Happy', file=nb_predictions_file)
  # for x_test, prediction in zip(X_test, nb_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=nb_predictions_file)

  # svm_predictions = svm_model.predict(X_test)
  # svm_predictions_file = open(predictions_path + 'svm_predictions.csv', 'w')
  # print('UserID,Happy', file=svm_predictions_file)
  # for x_test, prediction in zip(X_test, svm_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=svm_predictions_file)

  # rf_predictions = rf_model.predict(X_test)
  # rf_predictions_file = open(predictions_path + 'rf_predictions.csv', 'w')
  # print('UserID,Happy', file=rf_predictions_file)
  # for x_test, prediction in zip(X_test, rf_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=rf_predictions_file)



if __name__ == '__main__':
  main()
