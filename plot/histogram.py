import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
  filename_train = '../data/train.csv'
  df = pd.read_csv(filename_train)
  YOB = df['YOB']
  plt.title('Histogram for Year of Birth (YOB)')
  plt.hist(YOB.dropna(), bins='auto')
  plt.xlabel('Year of Birth (YOB)')
  plt.ylabel('Probability Density')
  plt.grid(True)
  plt.savefig('./output/histogram.png')



if __name__ == '__main__':
  main()
