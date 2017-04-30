import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  filename_train = '../data/train.csv'
  df = pd.read_csv(filename_train)
  df1 = df[['YOB','Income']].dropna()
  df1['Income'] = df['Income'].map({'under $25,000':12500,
                                   '$25,001 - $50,000':37500,
                                   '$50,000 - $74,999':62500,
                                   '$75,000 - $100,000':87500,
                                   '$100,001 - $150,000':125000,
                                   'over $150,000':175000})


  fig = plt.figure()
  ax = plt.subplot()
  plt.title('Relationship between YOB and Income')
  ax.scatter(df1['YOB'], df1['Income'], alpha=0.01)
  plt.xlabel('Year of Birth (YOB)')
  plt.ylabel('Income ($)')
  plt.savefig('./output/scatter_plot.png')



if __name__ == '__main__':
  main()
