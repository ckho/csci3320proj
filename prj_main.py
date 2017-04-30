from preprocess import transform
from preprocess import transform_for_test
from preprocess import fill_missing

from lr import LogisticRegression
from naive_bayes import NaiveBayes

from sklearn import svm
from sklearn import ensemble

def main():
  # load training data
  filename_train = './data/train.csv'
  train_dataset = transform(filename_train)
  X = train_dataset['data']
  y = train_dataset['target']

  # fill in missing data (optional)
  X_full = fill_missing(X, 'most_frequent', False)

  ### use the logistic regression
  print('Train the logistic regression classifier')
  lr_model = LogisticRegression()
  lr_model.fit(X, y)

  ### use the naive bayes
  print('Train the naive bayes classifier')
  nb_model = NaiveBayes()
  nb_model.fit(X, y)

  ## use the svm
  print('Train the SVM classifier')
  svm_model = svm.SVC(random_state=1)
  svm_model.fit(X, y)

  ## use the random forest
  print('Train the random forest classifier')
  rf_model = ensemble.RandomForestClassifier(random_state=1)
  rf_model.fit(X, y)

  ## get test data
  filename_train = './data/test.csv'
  test_dataset = transform_for_test(filename_test)
  X_test = test_dataset['data']

  # TO DO: Fill Missing

  predictions_path = './predictions/'

  ## do predictions
  lr_predictions = lr_model.predict(X_test)
  lr_predictions_file = open(predictions_path + 'lr_predictions.csv', 'w')
  print('UserID,Happy', file=lr_predictions_file)
  for x_test, prediction in zip(X_test, lr_predictions):
    print(str(int(x_test['UserID']))+','+str(int(prediction)), file=lr_predictions_file)

  nb_predictions = nb_model.predict(X_test)
  nb_predictions_file = open(predictions_path + 'nb_predictions.csv', 'w')
  print('UserID,Happy', file=nb_predictions_file)
  for x_test, prediction in zip(X_test, nb_predictions):
    print(str(int(x_test['UserID']))+','+str(int(prediction)), file=nb_predictions_file)

  svm_predictions = svm_model.predict(X_test)
  svm_predictions_file = open(predictions_path + 'svm_predictions.csv', 'w')
  print('UserID,Happy', file=svm_predictions_file)
  for x_test, prediction in zip(X_test, svm_predictions):
    print(str(int(x_test['UserID']))+','+str(int(prediction)), file=svm_predictions_file)

  rf_predictions = rf_model.predict(X_test)
  rf_predictions_file = open(predictions_path + 'rf_predictions.csv', 'w')
  print('UserID,Happy', file=rf_predictions_file)
  for x_test, prediction in zip(X_test, rf_predictions):
    print(str(int(x_test['UserID']))+','+str(int(prediction)), file=rf_predictions_file)



if __name__ == '__main__':
  main()
