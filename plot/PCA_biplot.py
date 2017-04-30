import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def transform(filename):
  """ preprocess the training data"""
  """ your code here """
  df = pd.read_csv(filename,
                   header=0,
                   index_col=0,
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

  data = df.drop('Happy', 1)
  target = df['Happy']
  return {'data':data,'target':target}



def main():
  filename_train = '../data/train.csv'
  train_dataset = transform(filename_train)
  X = train_dataset['data'].dropna()

  ## perform PCA
  n = len(X.columns)

  pca = PCA(n_components = n)
  # defaults number of PCs to number of columns in imported data (ie number of
  # features), but can be set to any integer less than or equal to that value

  pca.fit(X)

  ## project data into PC space

  # 0,1 denote PC1 and PC2; change values for other PCs
  xvector = pca.components_[0] # see 'prcomp(my_data)$rotation' in R
  yvector = pca.components_[1]

  xs = pca.transform(X)[:,0] # see 'prcomp(my_data)$x' in R
  ys = pca.transform(X)[:,1]


  fig = plt.figure(1, [20, 20])

  ## visualize projections

  ## Note: scale values for arrows and text are a bit inelegant as of now,
  ##       so feel free to play around with them

  for i in range(len(xvector)):
  # arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.2, head_length=0.2)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(X.columns.values)[i], color='r')

  for i in range(len(xs)):
  # circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo')

  plt.title('PCA Biplot')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.savefig('./output/PCA_biplot.png')


if __name__ == '__main__':
  main()