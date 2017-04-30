from preprocess import transform

import numpy as np
import pandas as pd
from naive_bayes import NaiveBayes


from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn.model_selection import train_test_split

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

  # numeric x
  numeric_cols = [ 'YOB', 'votes' ]
  x_num = X[numeric_cols].as_matrix()

  # scale to <0,1>

  x_max = np.amax(x_num, 0)

  x_num = x_num / x_max

  # ids = .UserID

  # categorical

  cat_X = X.drop(numeric_cols + ['UserID'], axis = 1)


  cat_X.fillna(0, inplace = True)


  x_cat = cat_X.T.to_dict().values()

  # vectorize

  vectorizer = DV(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  # complete x

  x_completed = np.hstack((x_num, vec_x_cat))


  X_train, X_verify, y_train, y_verify = train_test_split(x_completed, y, test_size=0.18, random_state=0)

  # X_train = np.dot(X_train, a)
  # X_verify = np.dot(X_verify, a)

  svm_model = svm.SVC(random_state=1, C=0.19, gamma=0.0028, shrinking=True, probability=True)
  svm_model.fit(X_train, y_train)
  print(svm_model.score(X_train, y_train))
  print(svm_model.score(X_verify, y_verify))

  # print('Train the naive bayes classifier')
  # print('Bernoulli')
  # nb_model = NaiveBayes(model='bernoulli')
  # nb_model.fit(X_train, y_train.values)
  # # nb_model.printPara()
  # print(nb_model.predict(X_verify))
  # print('Gaussian')
  # nb_model = NaiveBayes(model='gaussian')
  # nb_model.fit(X_train, y_train.values)
  # # nb_model.printPara()
  # print(nb_model.predict(X_verify))
  # print('Multinomial')
  # nb_model = NaiveBayes(model='multinomial')
  # nb_model.fit(X_train, y_train.values)
  # # nb_model.printPara()
  # print(nb_model.predict(X_verify))
  # print(nb_model.predict(X_verify))
  # print(nb_model.score(X_train, y_train.values))
  # print(nb_model.score(X_verify, y_verify.values))



if __name__ == '__main__':
  main()