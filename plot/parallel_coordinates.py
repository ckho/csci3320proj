# Male:
  # 'Female':0,
  # 'Male':1
# Income:
  # 'under $25,000':0,
  # '$25,001 - $50,000':1,
  # '$50,000 - $74,999':2,
  # '$75,000 - $100,000':3,
  # '$100,001 - $150,000':4,
  # 'over $150,000':5
# HouseholdStatus:
  # 'Domestic Partners (no kids)':0,
  # 'Domestic Partners (w/kids)':1,
  # 'Married (no kids)':2,
  # 'Married (w/kids)':3,
  # 'Single (no kids)':4,
  # 'Single (w/kids)':5
# EducationLevel:
  # 'Current K-12':0,
  # 'High School Diploma':1,
  # 'Current Undergraduate':2,
  # "Associate's Degree":3,
  # "Bachelor's Degree":4,
  # "Master's Degree":5,
  # 'Doctoral Degree':6
# Party:
  # 'Democrat':0,
  # 'Republican':1,
  # 'Independent':2,
  # 'Libertarian':3,
  # 'Other':4


import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import numpy as np

def main():
  filename_train = '../data/train.csv'
  df = pd.read_csv(filename_train)

  df1 = df[['Gender','Income', 'HouseholdStatus', 'EducationLevel', 'Party', 'Happy']].dropna()[:100]
  df1['Gender'] = df1['Gender'].map({'Female':0,
                                    'Male':1})
  df1['Income'] = df1['Income'].map({'under $25,000':0,
                                   '$25,001 - $50,000':1,
                                   '$50,000 - $74,999':2,
                                   '$75,000 - $100,000':3,
                                   '$100,001 - $150,000':4,
                                   'over $150,000':5})
  df1['HouseholdStatus'] = df1['HouseholdStatus'].map({'Domestic Partners (no kids)':0,
                                                     'Domestic Partners (w/kids)':1,
                                                     'Married (no kids)':2,
                                                     'Married (w/kids)':3,
                                                     'Single (no kids)':4,
                                                     'Single (w/kids)':5})
  df1['EducationLevel'] = df1['EducationLevel'].map({'Current K-12':0,
                                                   'High School Diploma':1,
                                                   'Current Undergraduate':2,
                                                   "Associate's Degree":3,
                                                   "Bachelor's Degree":4,
                                                   "Master's Degree":5,
                                                   'Doctoral Degree':6})
  df1['Party'] = df1['Party'].map({'Democrat':0,
                                 'Republican':1,
                                 'Independent':2,
                                 'Libertarian':3,
                                 'Other':4})

  df1['Happy'] = df1['Happy'].map({1:'Happy',
                                   0:'Unhappy'})

  fig = plt.figure()
  parallel_coordinates(df1, 'Happy', color=['r', 'b'])
  plt.title('Parallel Coordinates Plot')
  # plt.xlabel('Year of Birth (YOB)')
  # plt.ylabel('Income ($)')
  plt.savefig('./output/parallel_coordinates.png')



if __name__ == '__main__':
  main()
