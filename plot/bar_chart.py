import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  filename_train = '../data/train.csv'
  df = pd.read_csv(filename_train)
  df1 = df[['Happy','Income']].dropna()

  happy = df['Income'][df['Happy'].isin({1})].value_counts()
  unhappy = df['Income'][df['Happy'].isin({0})].value_counts()

  income = df['Income'].value_counts()

  n_groups = 6

  happy_for_chart = (happy['under $25,000']/income['under $25,000'],
                     happy['$25,001 - $50,000']/income['$25,001 - $50,000'],
                     happy['$50,000 - $74,999']/income['$50,000 - $74,999'],
                     happy['$75,000 - $100,000']/income['$75,000 - $100,000'],
                     happy['$100,001 - $150,000']/income['$100,001 - $150,000'],
                     happy['over $150,000']/income['over $150,000'])

  unhappy_for_chart = (unhappy['under $25,000']/income['under $25,000'],
                       unhappy['$25,001 - $50,000']/income['$25,001 - $50,000'],
                       unhappy['$50,000 - $74,999']/income['$50,000 - $74,999'],
                       unhappy['$75,000 - $100,000']/income['$75,000 - $100,000'],
                       unhappy['$100,001 - $150,000']/income['$100,001 - $150,000'],
                       unhappy['over $150,000']/income['over $150,000'])



  fig = plt.figure(1, [10, 10])
  ax = plt.subplot()
  index = np.arange(n_groups) * 1.4
  bar_width = 0.5
  opacity = 0.4
  error_config = {'ecolor': '0.3'}

  rects1 = plt.bar(index, happy_for_chart, bar_width,
                   alpha=opacity,
                   color='b',
                   error_kw=error_config,
                   label='Happy')

  rects2 = plt.bar(index + bar_width, unhappy_for_chart, bar_width,
                   alpha=opacity,
                   color='r',
                   error_kw=error_config,
                   label='Unhappy')

  plt.title('Relationship between Income and Happiness')
  plt.xlabel('Income ($)')
  plt.ylabel('Population')
  plt.xticks(index + bar_width / 2, ('under $25,000', '$25,001 - $50,000', '$50,000 - $74,999', '$75,000 - $100,000', '$100,001 - $150,000', 'over $150,000'))
  plt.legend()
  plt.tight_layout()
  plt.savefig('./output/bar_chart.png')



if __name__ == '__main__':
  main()
