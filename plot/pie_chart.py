import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  filename_train = '../data/train.csv'
  df = pd.read_csv(filename_train)
  labels = 'Happy', 'Unhappy'
  fig = plt.figure()
  # Men
  # df['Happy'][df['Gender'].isin({'Male'})]
  men = fig.add_subplot(1, 2, 1)
  plt.title('Men')
  happycount_men = df['Happy'][df['Gender'].isin({'Male'})].value_counts()
  sizes_men = [happycount_men[1], happycount_men[0]]
  men.pie(sizes_men, labels=labels, autopct='%1.1f%%', shadow=True)
  men.axis('equal')
  # Women
  women=fig.add_subplot(1, 2, 2)
  plt.title('Women')
  happycount_women = df['Happy'][df['Gender'].isin({'Female'})].value_counts()
  sizes_women = [happycount_women[1], happycount_women[0]]
  women.pie(sizes_women, labels=labels, autopct='%1.1f%%', shadow=True)
  women.axis('equal')

  plt.savefig('./output/pie_chart.png')



if __name__ == '__main__':
  main()
