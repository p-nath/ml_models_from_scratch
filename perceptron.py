import argparse
import numpy as np

def train(traindata, trainlabels, num_epochs, wts, target_error = 0.0):
	errors = []
	# ideally the algo will run till it converges i.e. error = 0
	for step in range(num_epochs):
		
		for i, label in zip(traindata, trainlabels):
			prediction = predict(i, wts)
			# any mislabels
			wts[1:] += (label - prediction) * i
			wts[0] += (label - prediction)

		errors.append(getPredictionAccuracy(traindata, trainlabels, wts))
		# print("Training error for Epoch[%d|%d]: %3.3f"%(step+1, num_epochs, getPredictionAccuracy(traindata, trainlabels, wts)))
		# print(np.mean(errors[-5:-2]))
		if errors[-1] == target_error or (len(errors) > 16 and np.mean(errors[-16:-2])+0.000005 >= errors[-1] >=  np.mean(errors[-16:-2])-0.000005):
			break
	# print(errors)
	# print("Mean training error over %d epochs: %3.3f"%(num_epochs, errors[-1]))


def predict(data, wts):
	# dot product as a similarity measure plus bias
	additive = np.dot(data, wts[1:]) + wts[0]
	# step function
	if additive > 0:
		return 1
	else:
		return 0

def getPredictionAccuracy(testdata, testlabels, wts):
	error_count = 0
	for datapoint, label in zip(testdata, testlabels):
		prediction = predict(datapoint, wts)
		if prediction != label:
			error_count += 1
	return error_count / len(testlabels)

def getBatches(dataset, k):
	# randomize the dataset before dividing
	np.random.shuffle(dataset)
	# dividing the dataset into k divisions
	# since the number of rows in the dataset may not be perfectly divisible by k
	# we find the extra number of rows 
	batch_size = len(dataset) // k
	extra = len(dataset) % k
	# divide into "k - extra" batches of size batch_size
	batches = np.split(dataset[:batch_size * (k - extra)], k - extra)
	# divide the rest into "extra" number of batches of size "batch_size+1"
	b = np.split(dataset[batch_size * (k - extra):], extra)
	for i in b:
		batches.append(i)
	return batches

def main():
	# parse commandline arguments
	args = ParseArgs()
	datafile = args.fin
	num_epochs = args.epochs
	mode = args.mode
	# read csv into numpy array
	# the first array row contain the feature names
	dataset = np.genfromtxt(datafile, delimiter=',')
	dataset = dataset[1:] # ignore the first row

	if mode == "erm":
		num_features = len(dataset[0]) - 1
		data = dataset[:, :-1]
		labels = dataset[:, -1]
		# initialize weights
		# +1 for bias
		wts = np.zeros(num_features + 1)
		train(data, labels, num_epochs, wts)
		print("Emperical risk measured on the training set: %3.3f / 1.000"%(getPredictionAccuracy(data, labels, wts)))
		print(wts)
	elif mode == "kfold":
		k = args.k
		batches = getBatches(dataset, k)

		num_features = len(dataset[0]) - 1
		wts = np.zeros(num_features + 1)
		errors = []
		for i in range(k):
			testset = batches[i][:, :-1]
			testlabels = batches[i][:, -1]
			trainset_size = len(dataset) - len(testset)
			trainset = []
			for batch in range(k):
				if batch != i:
					for row in batches[batch]:
						trainset.append(row)
			trainset = np.array(trainset)
			trainlabels = trainset[:, -1]
			trainset = trainset[:, :-1]
			# train on trainset
			train(trainset, trainlabels, num_epochs, wts)
			# predict and get error on testset
			errors.append(getPredictionAccuracy(testset, testlabels, wts))
			print("Error on %dth fold testing set: %3.3f / 1.000"%(i+1, errors[-1]))
		print("Mean Testing Error: %3.3f / 1.000"%(sum(errors)/k))
		print(wts)

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action="store", dest="fin", default=None,
                     help='Specify dataset.')
  parser.add_argument('--epochs', type=int, dest="epochs", default=100,
  										help='Specify number of epochs.')
  parser.add_argument('--mode', action="store", dest="mode", default="erm",
                     help='Specify mode,erm or kfold.')
  parser.add_argument('-k', type=int, dest="k", default="10",
  										help='Specify k for k fold cross validation.')
  return parser.parse_args()

if __name__ == "__main__":
	main()
