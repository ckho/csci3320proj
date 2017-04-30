import pandas as pd
import numpy as np
import scipy.stats as sp

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
  df['HouseholdStatus'] = df['HouseholdStatus'].map({'Domestic Partners (no kids)':1,
                                                     'Domestic Partners (w/kids)':2,
                                                     'Married (no kids)':3,
                                                     'Married (w/kids)':4,
                                                     'Single (no kids)':5,
                                                     'Single (w/kids)':6})
  df['EducationLevel'] = df['EducationLevel'].map({'Current K-12':1,
                                                   'High School Diploma':2,
                                                   'Current Undergraduate':3,
                                                   "Associate's Degree":4,
                                                   "Bachelor's Degree":5,
                                                   "Master's Degree":6,
                                                   'Doctoral Degree':7})
  df['Party'] = df['Party'].map({'Democrat':1,
                                 'Republican':2,
                                 'Independent':3,
                                 'Libertarian':4,
                                 'Other':5})
  df = df.apply(pd.to_numeric, errors='ignore')

  data = df.drop('Happy', 1).as_matrix()
  target = df['Happy'].as_matrix()
  return {'data':data,'target':target}

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


  return X_full
