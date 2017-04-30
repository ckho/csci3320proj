from preprocess import transform
from preprocess import transform_for_test
from preprocess import fill_missing

from lr import LogisticRegression
from naive_bayes import NaiveBayes

from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer

import numpy as np
import pandas as pd

def main():
  # load training data
  filename_train = './data/train.csv'
  train_dataset = transform(filename_train)
  X = train_dataset['data']
  y = train_dataset['target']

  X = X.replace({False:-1, True:1})

  X.loc[X.YOB < 1920, 'YOB'] = 0
  X.loc[X.YOB > 2004, 'YOB'] = 0
  X.loc[X.YOB.isnull(), 'YOB'] = 0

  numeric_cols = ['YOB', 'votes']

  x_num = X[numeric_cols].as_matrix()
  x_max = np.amax(x_num, 0)
  x_num = x_num / x_max

  cat_X = X.drop(numeric_cols + ['UserID'], axis = 1)
  cat_X.fillna(0, inplace = True)
  x_cat = cat_X.T.to_dict().values()

  # vectorize
  vectorizer = DV(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  # combine x
  x_completed = np.hstack((x_num, vec_x_cat))


  X_train, X_verify, y_train, y_verify = train_test_split(x_completed, y, test_size=0.18, random_state=0)

  # ### use the logistic regression
  print('Train the logistic regression classifier')
  lr_model = LogisticRegression()
  lr_model.fit(X_train, y_train)
  print(lr_model.score(X_verify, y_verify))

  print('Train the logistic regression classifier-sklearn')
  lr_model = lr2()
  lr_model.fit(X_train, y_train)
  print(lr_model.score(X_verify, y_verify))

  # # ### use the naive bayes
  print('Train the naive bayes classifier')
  nb_model = NaiveBayes(model='gaussian')
  nb_model.fit(X_train, y_train)
  print(nb_model.score(X_verify, y_verify))

  print('Train the naive bayes classifier-sklearn')
  nb_model = GaussianNB()
  nb_model.fit(X_train, y_train)
  print(nb_model.score(X_verify, y_verify))

  ## use the svm
  print('Train the SVM classifier')
  svm_model = svm.SVC(random_state=1, C=0.19, gamma=0.0028, shrinking=True, probability=True)
  svm_model.fit(X_train, y_train)
  print(svm_model.score(X_verify, y_verify))

  ## use the random forest
  print('Train the random forest classifier')
  rf_model = ensemble.RandomForestClassifier(random_state=1, n_estimators=4300, n_jobs=-1)
  rf_model.fit(X_train, y_train)
  print(rf_model.score(X_verify, y_verify))


  # ## get test data
  # filename_train = './data/test.csv'
  # test_dataset = transform_for_test(filename_test)
  # X_test = test_dataset['data']

  # X_test = X_test.replace({False:-1, True:1})

  # X_test.loc[X_test.YOB < 1920, 'YOB'] = 0
  # X_test.loc[X_test.YOB > 2004, 'YOB'] = 0
  # X_test.loc[X_test.YOB.isnull(), 'YOB'] = 0

  # numeric_cols = ['YOB', 'votes']

  # x_test_num = X_test[numeric_cols].as_matrix()
  # x_test_num = x_test_num / x_max

  # cat_X_test = X_test.drop(numeric_cols + ['UserID'], axis = 1)
  # cat_X_test.fillna(0, inplace = True)
  # x_test_cat = cat_X_test.T.to_dict().values()

  # # vectorize
  # vec_x_test_cat = vectorizer.transform(x_test_cat)

  # # combine x
  # x_test_completed = np.hstack((x_test_num, vec_x_test_cat))

  # # # TO DO: Fill Missing

  # predictions_path = './predictions/'

  # ## do predictions
  # lr_predictions = lr_model.predict(x_test_completed)
  # lr_predictions_file = open(predictions_path + 'lr_predictions.csv', 'w')
  # print('UserID,Happy', file=lr_predictions_file)
  # for x_test, prediction in zip(x_test_completed, lr_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=lr_predictions_file)

  # nb_predictions = nb_model.predict(x_test_completed)
  # nb_predictions_file = open(predictions_path + 'nb_predictions.csv', 'w')
  # print('UserID,Happy', file=nb_predictions_file)
  # for x_test, prediction in zip(x_test_completed, nb_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=nb_predictions_file)

  # svm_predictions = svm_model.predict(x_test_completed)
  # svm_predictions_file = open(predictions_path + 'svm_predictions.csv', 'w')
  # print('UserID,Happy', file=svm_predictions_file)
  # for x_test, prediction in zip(x_test_completed, svm_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=svm_predictions_file)

  # rf_predictions = rf_model.predict(x_test_completed)
  # rf_predictions_file = open(predictions_path + 'rf_predictions.csv', 'w')
  # print('UserID,Happy', file=rf_predictions_file)
  # for x_test, prediction in zip(x_test_completed, rf_predictions):
  #   print(str(int(x_test['UserID']))+','+str(int(prediction)), file=rf_predictions_file)

#just testing git

if __name__ == '__main__':
  main()
