from preprocess import transform_for_lr, transform_for_lr_test
from preprocess import transform_for_svm, transform_for_svm_test
from preprocess import transform_for_nb, transform_for_nb_test
from preprocess import transform_for_rf, transform_for_rf_test
from preprocess import fill_missing

from lr import LogisticRegression
from naive_bayes import NaiveBayes

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr2
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
from numpy import random

import time

def main():
  # load training data
  filename_train = './data/train.csv'

  ## use the logistic regression
  lr_train_dataset = transform_for_lr(filename_train)
  lr_X_train, lr_X_verify11, lr_y_train, lr_y_verify11 = train_test_split(lr_train_dataset['data'], lr_train_dataset['target'], test_size=0.1, random_state=100)
  lr_X_train11, lr_X_verify, lr_y_train11, lr_y_verify = train_test_split(lr_train_dataset['data'], lr_train_dataset['target'], test_size=0.18, random_state=0)
  print('Train the logistic regression classifier')
  t1 = time.time()
  lr_model = LogisticRegression()
  lr_model.fit(lr_X_train, lr_y_train)
  lr_accuracy = lr_model.score(lr_X_verify11, lr_y_verify11)
  print('Accuracy: ' + str(lr_accuracy))
  print('Runtime: ' + str(time.time() - t1))

  print('Train the logistic regression classifier-sklearn')
  t1 = time.time()
  lr2_model = lr2()
  lr2_model.fit(lr_X_train, lr_y_train)
  lr2_accuracy = lr2_model.score(lr_X_verify11, lr_y_verify11)
  print('Accuracy: ' + str(lr2_accuracy))
  print('Runtime: ' + str(time.time() - t1))

  ## use the naive bayes
  nb_train_dataset = transform_for_nb(filename_train)
  nb_X_train, nb_X_verify11, nb_y_train, nb_y_verify11 = train_test_split(nb_train_dataset['data'], nb_train_dataset['target'], test_size=0.02, random_state=20)
  nb_X_train11, nb_X_verify, nb_y_train11, nb_y_verify = train_test_split(nb_train_dataset['data'], nb_train_dataset['target'], test_size=0.18, random_state=0)

  print('Train the naive bayes classifier-MultinomialNB')
  t1 = time.time()
  nb_model = NaiveBayes(model='multinomial')
  nb_model.fit(nb_X_train+1, nb_y_train)
  nb_accuracy = nb_model.score(nb_X_verify11, nb_y_verify11)
  print('Accuracy: ' + str(nb_accuracy))
  print('Runtime: ' + str(time.time() - t1))

  print('Train the naive bayes classifier-sklearn-MultinomialNB')
  t1 = time.time()
  nb2_model = MultinomialNB()
  nb2_model.fit(nb_X_train+1, nb_y_train)
  nb2_accuracy = nb2_model.score(nb_X_verify11, nb_y_verify11)
  print('Accuracy: ' + str(nb2_accuracy))
  print('Runtime: ' + str(time.time() - t1))


  ## use the svm
  svm_train_dataset = transform_for_svm(filename_train)
  svm_X_train, svm_X_verify, svm_y_train, svm_y_verify = train_test_split(svm_train_dataset['data'], svm_train_dataset['target'], test_size=0.18, random_state=0)
  print('Train the SVM classifier')
  t1 = time.time()
  svm_model = svm.SVC(random_state=1, C=0.19, gamma=0.0028, shrinking=True, probability=True)
  svm_model.fit(svm_X_train, svm_y_train)
  svm_accuracy = svm_model.score(svm_X_verify, svm_y_verify)
  print('Accuracy: ' + str(svm_accuracy))
  print('Runtime: ' + str(time.time() - t1))

  ## use the random forest
  rf_train_dataset = transform_for_rf(filename_train)
  rf_X_train, rf_X_verify, rf_y_train, rf_y_verify = train_test_split(rf_train_dataset['data'], rf_train_dataset['target'], test_size=0.18, random_state=0)

  print('Train the random forest classifier')
  t1 = time.time()
  rf_model = ensemble.RandomForestClassifier(random_state=1, n_estimators=4300, n_jobs=-1)
  rf_model.fit(rf_X_train, rf_y_train)
  rf_accuracy = rf_model.score(rf_X_verify, rf_y_verify)
  print('Accuracy: ' + str(rf_accuracy))
  print('Runtime: ' + str(time.time() - t1))


  ## try voting classification
  print('Try voting classification by 4 classifiers')
  lr_result = lr_model.predict(lr_X_verify)
  nb_result = nb_model.predict(nb_X_verify)
  svm_result = svm_model.predict(svm_X_verify)
  rf_result = rf_model.predict(rf_X_verify)

  probability = [lr_accuracy, nb_accuracy, svm_accuracy, rf_accuracy]
  probability = probability / sum(probability)
  print(probability*4)

  result = []
  random.seed = 0
  for a, b, c, d in zip(lr_result, nb_result, svm_result, rf_result):
    count = 0
    if a == 1:
      count += probability[0]*4
    if b == 1:
      count += probability[1]*4
    if c == 1:
      count += probability[2]*4
    if d == 1:
      count += probability[3]*4
    if count > 2:
      result.append(1)
    elif count < 2:
      result.append(0)
    else:
      # result.append(random.choice([a, b, c, d]))
      result.append(random.choice([a, b, c, d], p=probability))

  accuracy = sum(result == lr_y_verify) / len(lr_y_verify)
  print('Accuracy: ' + str(accuracy))


  ## get test data
  filename_test = './data/test.csv'
  df = pd.read_csv(filename_test)



  predictions_path = './predictions/'

  ## do predictions

  lr_test_dataset = transform_for_lr_test(filename_test)
  X_lr_test = lr_test_dataset['data']
  lr_predictions = lr_model.predict(X_lr_test)
  lr_predictions_file = open(predictions_path + 'lr_predictions.csv', 'w')
  print('UserID,Happy', file=lr_predictions_file)
  for x_test, prediction in zip(df.values, lr_predictions):
    print(str(int(x_test[0]))+','+str(int(prediction)), file=lr_predictions_file)


  nb_test_dataset = transform_for_nb_test(filename_test)
  X_nb_test = nb_test_dataset['data']
  nb_predictions = nb_model.predict(X_nb_test)
  nb_predictions_file = open(predictions_path + 'nb_predictions.csv', 'w')
  print('UserID,Happy', file=nb_predictions_file)
  for x_test, prediction in zip(df.values, nb_predictions):
    print(str(int(x_test[0]))+','+str(int(prediction)), file=nb_predictions_file)

  svm_test_dataset = transform_for_svm_test(filename_test)
  X_svm_test = svm_test_dataset['data']
  svm_predictions = svm_model.predict(X_svm_test)
  svm_predictions_file = open(predictions_path + 'svm_predictions.csv', 'w')
  print('UserID,Happy', file=svm_predictions_file)
  for x_test, prediction in zip(df.values, svm_predictions):
    print(str(int(x_test[0]))+','+str(int(prediction)), file=svm_predictions_file)

  rf_test_dataset = transform_for_rf_test(filename_test)
  X_rf_test = rf_test_dataset['data']
  rf_predictions = rf_model.predict(X_rf_test)
  rf_predictions_file = open(predictions_path + 'rf_predictions.csv', 'w')
  print('UserID,Happy', file=rf_predictions_file)
  for x_test, prediction in zip(df.values, rf_predictions):
    print(str(int(x_test[0]))+','+str(int(prediction)), file=rf_predictions_file)

  vote_predictions = []
  random.seed = 0
  for a, b, c, d in zip(lr_predictions, nb_predictions, svm_predictions, rf_predictions):
    count = 0
    if a == 1:
      count += probability[0]*4
    if b == 1:
      count += probability[1]*4
    if c == 1:
      count += probability[2]*4
    if d == 1:
      count += probability[3]*4
    if count > 2:
      vote_predictions.append(1)
    elif count < 2:
      vote_predictions.append(0)
    else:
      vote_predictions.append(random.choice([a, b, c, d], p=probability))

  vote_predictions_file = open(predictions_path + 'vote_predictions.csv', 'w')
  for x_test, prediction in zip(df.values, vote_predictions):
    print(str(int(x_test[0]))+','+str(int(prediction)), file=vote_predictions_file)


if __name__ == '__main__':
  main()
