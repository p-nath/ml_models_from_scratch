import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def createKmeansClusters(data, labels, k, distance_func, num_epochs):
  # randomly choose centroid
  x = [np.random.randint(0, len(data)-1) for i in range(k)]
  centroids = [data[i] for i in x]
  old_centroids = None
  C = []
  clusters = []
  converged = 0
  n = 0

  while n < num_epochs and converged != 5:
    C = [[] for i in range(k)]
    clusters = [[] for i in range(k)]
    distances = []
    for pt_i in range(len(data)):
      pt = data[pt_i]
      distances = [[c_i, distance_func(pt, centroids[c_i])] for c_i in range(len(centroids))]
      # sort in ascending order according to distance
      distances.sort(key=lambda x:x[1])
      C[distances[0][0]].append(pt)
      clusters[distances[0][0]].append(pt_i)

    for i in range(k):
      centroids[i] = np.sum(C[i], axis=0)/ len(C[i])

    # convergence condition
    if old_centroids:
      checked = 0
      centroid_dist = [distance_func(centroids[i], old_centroids[i]) for i in range(k)]
      for i in centroid_dist:
        if 0.01 >= i >= -0.01:
          checked += 1
        else:
          break
      if checked == k:
        converged += 1
      else:
        converged = 0
      # print(centroid_dist)

    old_centroids = [i for i in centroids]
    n += 1

  if converged == 5:
    print("Model converged after %d iterations!"%(n))
  else:
    print("Model finished %d iterations."%(num_epochs))
  return centroids, clusters

def drawClusters(data, clusters):
  pca = PCA(n_components=2)
  reduced_data = pca.fit_transform(data)
  colors = ['darkorange', 'darkcyan', 'darkgreen', 'darkred', 'darkblue']
  if len(clusters) >len(colors):
    colors = [np.random.rand(3,) for i in range(len(clusters))]

  for c_id in range(len(clusters)):
    plt.scatter(reduced_data[clusters[c_id],0], reduced_data[clusters[c_id],1], c=colors[c_id], alpha=0.4, s=20)
  plt.show()

def getMetrics(predicted_clusters, labels):
  count = 0
  # positive_counts
  label_set = set(labels)
  known_clusters = [set() for i in label_set]
  for i in range(len(labels)):
    known_clusters[int(labels[i])].add(i)

  for kc_i in range(len(known_clusters)):
    for pc_i in range(len(predicted_clusters)):
      count = 0
      for point in predicted_clusters[pc_i]:
        if point in known_clusters[kc_i]:
          count += 1

      print("Predicted cluster %d accounts for %3.2f(%d) of datapoints in cluster labelled %d."
        %(pc_i, (count*100)/len(known_clusters[kc_i]), count, kc_i))
      # print("Positive diagnosis %3.2f"
      #   %((count*100)/len(predicted_clusters[pc_i])))

def euclideanDist(x, y):
  dist = 0
  for i, j  in zip(x, y):
    dist += (i - j) ** 2
  return np.sqrt(dist)

def manhattanDist(x, y):
  dist = 0
  for i, j  in zip(x, y):
    dist += abs(i - j)
  return dist

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action="store", dest="fin", default=None,
                     help='Specify dataset.')
  parser.add_argument('-k', type=int, dest="k", default=2,
  										help='Specify k for k-nearest_neighbours.')
  parser.add_argument('--distance_metric', action="store", dest="distance_metric", default="euclidean",
                     help='Specify distance metric type: euclidean or manhattan.')
  parser.add_argument('--num_epochs', type=int, dest="n", default=5,
                      help='Specify number of epochs.')
  parser.add_argument('--plot', action="store_true", dest="plot", 
                      help='Metion if you want to see the predicted clusters visualized.')
  parser.add_argument('--scale', action="store_true", dest="scale", 
                      help='Metion if you want to scale data.')
  return parser.parse_args()

def main():
  # parse commandline arguments
  args = parseArgs()
  datafile = args.fin
  k = args.k
  distance_metric = args.distance_metric
  n = args.n

  if distance_metric == "euclidean":
    distance_func = euclideanDist
  elif distance_metric == "manhattan":
    distance_func = manhattanDist

  # read csv into numpy array
  dataset = np.genfromtxt(datafile, delimiter=',')
  dataset = dataset[1:] # ignore the first row of feature names

  np.random.shuffle(dataset)
  data = dataset[:, :-1]
  labels = dataset[:, -1]

  if args.scale:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

  centres, clusters = createKmeansClusters(data, labels, k, distance_func, n)
  getMetrics(clusters, labels)

  if args.plot:
    drawClusters(data,clusters)

if __name__ == "__main__":
  main()