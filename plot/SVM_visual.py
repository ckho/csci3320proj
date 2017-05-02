import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm


def main():
  filename_train = '../data/train.csv'

  df = pd.read_csv(filename_train,
                   header=0)
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})

  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})

  df.fillna(0, inplace = True)

  X = df[['Income', 'EducationLevel']]
  y = df['Happy']

  X = X.values
  y = y.values

  print('Train the SVM classifier')
  svm_model = svm.SVC(random_state=1, C=0.19, gamma=0.0028, shrinking=True, probability=True)
  svm_model.fit(X, y)

  h = .02  # step size in the mesh

  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


  Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.xticks(())
  plt.yticks(())

  plt.title('SVC with RBF kernel')
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.savefig('./output/SVM_visual.png')


if __name__ == '__main__':
  main()