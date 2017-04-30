import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import normalize

class LogisticRegression:
  def __init__(self):
    return None


  def _sigmoid(self,z):
    return 1.0 / (1.0 + np.exp(-z))


  def _cost(self, theta):
    temp = self._sigmoid(np.dot(self.X, theta))
    first = -1 * self.y * np.log(temp)
    second = (1 - self.y) * np.log(1 - temp)
    return np.average(first - second)


  def _gradientDescend(self, theta):
    error = self._sigmoid(np.dot(self.X, theta)) - self.y
    return np.dot(error.T, self.X)


  def fit(self, X, y):
    self.X = normalize(X, norm='l2')
    self.y = y
    self.theta = np.zeros(X.shape[1])
    result = opt.fmin_l_bfgs_b(self._cost, x0=self.theta, fprime=self._gradientDescend, disp=0)
    self.theta = result[0]
    return self


  def predict(self, X):
    X = normalize(X, norm='l2')
    p = self._sigmoid(np.dot(X, self.theta))
    return np.where(p >= .5, 1, 0)


  def score(self, X, y):
    return sum(self.predict(X) == y) / len(y)
