import numpy as np
from sklearn.preprocessing import normalize



class NaiveBayes:
  def __init__(self, alpha=1.0, model='gaussian'):
    self.alpha = alpha # Not applicable to gaussian model
    self.model = model


  def fit(self, X, y):
    separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

    if self.model == 'gaussian':
      self.gaussian_model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                      for i in separated])
    else:
      count_sample = X.shape[0]
      self.class_log_prior = [np.log(len(i) / count_sample) for i in separated]
      count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha

      if self.model == 'multinomial':
        self.feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)
      elif self.model == 'bernoulli':
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob = count / n_doc[np.newaxis].T
    return self



  def _prob(self, x, mean, std):
    exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
    return np.log(exponent / (np.sqrt(2 * np.pi) * std))


  def _predict_log_proba(self, X):
    if self.model == 'multinomial':
      array = [(self.feature_log_prob * x).sum(axis=1) + self.class_log_prior for x in X]
    elif self.model == 'bernoulli':
      array = [(np.log(self.feature_prob) * x + \
                np.log(1 - self.feature_prob) * np.abs(x - 1)
               ).sum(axis=1) + \
               self.class_log_prior for x in X]
    elif self.model == 'gaussian':
      array = [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.gaussian_model] for x in X]
    return array


  def predict(self, X):
    return np.argmax(self._predict_log_proba(X), axis=1)


  def score(self, X, y):
    return sum(self.predict(X) == y) / len(y)
