import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

def predictKNN(trainData, trainLabels, testData, k):
  distances = []
  for i in range(len(trainData)):
    distances.append([trainLabels[i], euclideanDist(testData, trainData[i])])

  distances.sort(key=lambda x:x[1])

  label = [[i, 0] for i in set(trainLabels)]

  for d_i in range(k):
    label[int(distances[d_i][0])][1] += 1


  label.sort(key=lambda x:x[1])
  return label[-1][0]

def getKNNAccuracy(trainData, trainLabels, testData, testLabels, k):
  labels = []
  for i in testData:
    labels.append(predictKNN(trainData, trainLabels, i, k))
  count = 0
  for i in range(len(testData)):
    if testLabels[i] == labels[i]:
      count += 1
  return count/len(testData)

def euclideanDist(x, y):
  dist = 0
  for i, j  in zip(x, y):
    dist += (i - j) ** 2
  return np.sqrt(dist)

def splitData(dataset, seed=None):
  n = int(0.8 * len(dataset))
  if seed:
    np.random.seed(seed)
  np.random.shuffle(dataset)
  return dataset[:n], dataset[n:]

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action="store", dest="fin", default=None,
                     help='Specify dataset.')
  parser.add_argument('-k', type=int, dest="k", default=10,
  										help='Specify k for k-nearest_neighbours.')
  parser.add_argument('--seed', type=int, dest="seed", default=None,
                      help='Specify seed.')
  return parser.parse_args()

def main():
  # parse commandline arguments
  args = parseArgs()
  datafile = args.fin
  k = args.k
  seed = args.seed

  # read csv into numpy array
  dataset = np.genfromtxt(datafile, delimiter=',')
  dataset = dataset[1:] # ignore the first row of feature names

  # split data into trainset and test set in 80:20 ratio
  trainSet, testSet = splitData(dataset, seed)
  trainData = trainSet[:, :-1]
  trainLabels = trainSet[:, -1]
  testData = testSet[:, :-1]
  testLabels = testSet[:, -1]

  accuracy = getKNNAccuracy(trainData, trainLabels, testData, testLabels, k)

  print("Accuracy:", accuracy*100, "%")

if __name__ == "__main__":
  main()