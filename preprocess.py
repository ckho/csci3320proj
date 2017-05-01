import pandas as pd
import numpy as np
import scipy.stats as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

def transform(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Single (no kids)':1,
                                                     'Single (w/kids)':2,
                                                     'Domestic Partners (no kids)':3,
                                                     'Domestic Partners (w/kids)':4,
                                                     'Married (no kids)':5,
                                                     'Married (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Independent':2,
                                 'Other':3,
                                 'Libertarian':4,
                                 'Republican':5})
  # df['Age'] = 2014 - df['YOB']
  # df = df.replace({False:-1, True:1})
  # df = df.apply(pd.to_numeric, errors='ignore')

  data = df.drop('Happy', 1).values
  target = df['Happy'].values
  return {'data':data,'target':target}


def transform_lg_test(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Single (no kids)':1,
                                                     'Single (w/kids)':2,
                                                     'Domestic Partners (no kids)':3,
                                                     'Domestic Partners (w/kids)':4,
                                                     'Married (no kids)':5,
                                                     'Married (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Independent':2,
                                 'Other':3,
                                 'Libertarian':4,
                                 'Republican':5})

  # df['Age'] = 2014 - df['YOB']
  # df = df.replace({False:-1, True:1})
  # df = df.apply(pd.to_numeric, errors='ignore')

  X = df

  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'mean',0)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

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
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed}


def transform_for_lg(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Single (no kids)':1,
                                                     'Single (w/kids)':2,
                                                     'Domestic Partners (no kids)':3,
                                                     'Domestic Partners (w/kids)':4,
                                                     'Married (no kids)':5,
                                                     'Married (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Independent':2,
                                 'Other':3,
                                 'Libertarian':4,
                                 'Republican':5})
  # df['Age'] = 2014 - df['YOB']
  # df = df.replace({False:-1, True:1})
  # df = df.apply(pd.to_numeric, errors='ignore')


  X = df.drop('Happy', 1)
  y = df['Happy']
  X = X.replace({False:-1, True:1})

  X_filled = fill_missing(X.values,'mean',0)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

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
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed,'target':y}

def transform_for_nb_test(filename):
  X = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])

  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'none',1)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

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
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  vec_x_cat = vec_x_cat+1

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed}

def transform_for_nb(filename):
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  X = df.drop('Happy', 1)
  y = df['Happy']

  X = X.replace({False:-1, True:1})

  X_filled = fill_missing(X.values,'none',1)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

  X['YOB'] = X['YOB'].astype(float)
  X['votes'] = X['votes'].astype(float)

  X.loc[X.YOB < 1920, 'YOB'] = 0
  X.loc[X.YOB > 2004, 'YOB'] = 0
  X.loc[X.YOB.isnull(), 'YOB'] = 0

  numeric_cols = ['YOB', 'votes']

  x_num = X[numeric_cols].as_matrix()
  x_max = np.amax(x_num, 0)
  x_num = x_num / x_max

  x_num = np.round(x_num * 6)

  cat_X = X.drop(numeric_cols + ['UserID'], axis = 1)
  cat_X.fillna(0, inplace = True)
  x_cat = cat_X.T.to_dict().values()

  # vectorize
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  vec_x_cat = vec_x_cat+1

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed,'target':y}

def transform_for_svm_test(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Single (no kids)':1,
                                                     'Single (w/kids)':2,
                                                     'Domestic Partners (no kids)':3,
                                                     'Domestic Partners (w/kids)':4,
                                                     'Married (no kids)':5,
                                                     'Married (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Independent':2,
                                 'Other':3,
                                 'Libertarian':4,
                                 'Republican':5})

  # df['Age'] = 2014 - df['YOB']
  # df = df.replace({False:-1, True:1})
  # df = df.apply(pd.to_numeric, errors='ignore')

  X = df

  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'none',1)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

  X.loc[X.YOB < 1920, 'YOB'] = 0
  X.loc[X.YOB > 2004, 'YOB'] = 0
  X.loc[X.YOB.isnull(), 'YOB'] = 0

  numeric_cols = ['YOB', 'votes']

  x_num = X[numeric_cols].as_matrix()
  x_max = np.amax(x_num, 0)
  x_num = x_num / x_max

  x_num = (x_num * 6).round()

  cat_X = X.drop(numeric_cols + ['UserID'], axis = 1)
  cat_X.fillna(0, inplace = True)
  x_cat = cat_X.T.to_dict().values()

  # vectorize
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed}


def transform_for_svm(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  df['Income'] = df['Income'].map({'under $25,000':1,
                                   '$25,001 - $50,000':2,
                                   '$50,000 - $74,999':3,
                                   '$75,000 - $100,000':4,
                                   '$100,001 - $150,000':5,
                                   'over $150,000':6})
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Single (no kids)':1,
                                                     'Single (w/kids)':2,
                                                     'Domestic Partners (no kids)':3,
                                                     'Domestic Partners (w/kids)':4,
                                                     'Married (no kids)':5,
                                                     'Married (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   "Associate's Degree":3,
                                                   'Current Undergraduate':4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Independent':2,
                                 'Other':3,
                                 'Libertarian':4,
                                 'Republican':5})
  # df['Age'] = 2014 - df['YOB']
  # df = df.replace({False:-1, True:1})
  # df = df.apply(pd.to_numeric, errors='ignore')

  X = df.drop('Happy', 1)
  y = df['Happy']

  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'most_frequent',1)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

  X.loc[X.YOB < 1920, 'YOB'] = 0
  X.loc[X.YOB > 2004, 'YOB'] = 0
  X.loc[X.YOB.isnull(), 'YOB'] = 0

  numeric_cols = ['YOB', 'votes']

  x_num = X[numeric_cols].as_matrix()
  x_max = np.amax(x_num, 0)
  x_num = x_num / x_max

  x_num = (x_num * 6).round()

  cat_X = X.drop(numeric_cols + ['UserID'], axis = 1)
  cat_X.fillna(0, inplace = True)
  x_cat = cat_X.T.to_dict().values()

  # vectorize
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)

  x_completed = np.hstack((x_num, vec_x_cat))

  return {'data':x_completed,'target':y}


def transform_for_rf_test(filename):
  X = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'medium',0)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

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
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)
  vec_x_cat = vec_x_cat + 1
  encoder = OneHotEncoder(sparse = False)
  encoder.fit(vec_x_cat)
  enc_x_cat = encoder.transform(vec_x_cat)

  x_completed = np.hstack((x_num, enc_x_cat))

  return {'data':x_completed}


def transform_for_rf(filename):
  df = pd.read_csv(filename,
                   header=0,
                   # index_col=0,
                   true_values=['Yes','Check!','Yes!','1',
                                'Male','Public','Science',
                                'Study first','Giving',
                                'Idealist','Standard hours',
                                'Hot headed','Happy','A.M.',
                                'Circumstances','Start','TMI',
                                'People','Tunes','Supportive',
                                'Mac','Cautious','Socialize',
                                'Online','Yay people!','Rent',
                                'Mom','Optimist'],
                   false_values=['No', 'Only-child','Nope',
                                 'Umm...','0','Female','Private',
                                 'Art','Try first','Receiving',
                                 'Pragmatist','Odd hours',
                                 'Cool headed','Right','P.M.',
                                 'Me','End','Mysterious','Technology',
                                 'Talk','Demanding','PC','Risk-friendly',
                                 'Space','In-person','Grrr people',
                                 'Own','Dad','Pessimist'],
                   na_values=['NA'])
  X = df.drop('Happy', 1)
  y = df['Happy']

  X = X.replace({False:-1, True:1})
  X_filled = fill_missing(X.values,'medium',0)
  X = pd.DataFrame(data=X_filled, columns=X.columns)

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
  vectorizer = DictVectorizer(sparse = False)
  vec_x_cat = vectorizer.fit_transform(x_cat)
  vec_x_cat = vec_x_cat + 1
  encoder = OneHotEncoder(sparse = False)
  encoder.fit(vec_x_cat)
  enc_x_cat = encoder.transform(vec_x_cat)

  x_completed = np.hstack((x_num, enc_x_cat))

  return {'data':x_completed,'target':y}


def fill_missing(X, strategy, isClassified):
  """
   @X: input matrix with missing data filled by nan
   @strategy: string, 'median', 'mean', 'most_frequent'
   @isclassfied: boolean value, if isclassfied == true, then you need build a
   decision tree to classify users into different classes and use the
   median/mean/mode values of different classes to fill in the missing data;
   otherwise, just take the median/mean/most_frequent values of input data to
   fill in the missing data
  """
  if not(isClassified):
    for col in range(X.shape[1]):
      replacement = 0
      if strategy == 'median':
        replacement = np.nanmedian(X[:,col])
      elif strategy == 'mean':
        replacement = np.nanmean(X[:,col])
      elif strategy == 'most_frequent':
        replacement = sp.stats.mode(X[:,col], nan_policy='omit')

      for row in range(X[:,col].shape[0]):
        if np.isnan(X[row,col]):
          X[row,col] = replacement

  # if(isClassified):


  print(X)





  return X
