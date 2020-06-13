import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

class DecisionStump():
	def __init__(self, threshold, feature_index):
		self.threshold = threshold
		self.feature_index = feature_index
		self.contribution = None

	def predict(self, data):
		predictions = np.ones(len(data))
		for i in range(len(data)):
			if data[i][self.feature_index] < self.threshold:
				predictions[i] = -1
		return predictions

	def __str__(self):
		s = "sign - " + str(self.sign)
		s += "\nthreshold - " + str(self.threshold)
		s += "\nfeature index - " + str(self.feature_index) 
		s += "\ncontribution - " + str(self.contribution)
		return s

def getBestStump(dataset):
	min_error = float('inf')
	num_features = len(dataset[0]) - 2

	# select stump with least error
	for feature in range(num_features):
		# sort the array [data, labels, wts]
		dataset = dataset[np.argsort(dataset[:, feature])]
		# extract the data, labels and wts
		data = dataset[:, :-2]
		labels = dataset[:, -2]
		wts = dataset[:, -1]
		feature_col = dataset[:, feature]

		error = sum(wts[labels == 1])
		if error < min_error:
			min_error = error
			theta = data[0][feature] - 1
			feature_index = feature

		for i in range(len(data)):
			error = error - labels[i] * wts[i]
			# if error < min_error save the config
			if error < min_error and i + 1 < len(feature_col) and feature_col[i] != feature_col[i+1]:
				min_error = error
				theta = 0.5 * (feature_col[i] + feature_col[i+1])
				feature_index = feature

	best_stump = DecisionStump(theta, feature_index)
	return best_stump


# Adaboost functions
def train(traindata, trainlabels, T, classifiers):
	num_features = len(traindata[0])
	pred_train = np.zeros(len(traindata))
	num_rows = len(traindata)
	wts = np.full(num_rows, 1/num_rows)
	for i in range(T):
		# get the best stump with the minimum error T times
		decision_stump = getBestStump(np.column_stack((traindata, trainlabels, wts)))
		predictions = decision_stump.predict(traindata)
		error = sum(wts[predictions != trainlabels])
		decision_stump.contribution = 0.5 * math.log(1/error - 1) 
		# if error
		# wts = np.multiply(wts, np.exp([float(x)*decision_stump.contribution for x in miss2]))
		wts = np.multiply(wts, np.exp(-decision_stump.contribution*trainlabels*predictions))
		# normalize the weights 
		wts = wts / np.sum(wts)
		# print(calculateError(predictions, trainlabels))
		classifiers.append(decision_stump)
	return wts

def calculateError(predictions, labels):
	return sum(predictions != labels) / len(labels)

def predict(testdata, classifiers):
	final_predicted_labels = np.zeros(len(testdata))

	# add contribution of each classifier stump
	for classifier in classifiers:
		stump_predictions = classifier.predict(testdata)
		final_predicted_labels += classifier.contribution * stump_predictions
	
	# return array of signs
	return np.sign(final_predicted_labels)

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
	mode = args.mode
	T = args.t

	# read csv into numpy array
	# the first array row contain the feature names so ignore that
	dataset = np.genfromtxt(datafile, delimiter=',')
	dataset = dataset[1:] # ignore the first row
	for i in range(len(dataset[:, -1])):
		if dataset[i][-1] == 0:
			dataset[i][-1] = -1

	if mode == "erm":
		data = dataset[:, :-1]
		labels = dataset[:, -1]
	
		classifiers = []
		# train on train set and get the hypothesis i.e. updated wts
		wts = train(data, labels, T, classifiers)
		# predict on trainset and calculate error
		predicted_labels = predict(data, classifiers)
		print("Emperical Risk measured on training set:%3.3f / 1.00"%(calculateError(predicted_labels, labels)))
		print("The output hypothesis:\n", wts)
	elif mode == "kfold":
		k = args.k
		batches = getBatches(dataset, k)
		errors = []

		for i in range(k):
			# divide into testingset and training set
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
			
			classifiers = []
			# train adaboost
			wts = train(trainset, trainlabels, T, classifiers)
			# predict labels of testset and calculate error on test set
			predicted_labels = predict(testset, classifiers)
			errors.append(calculateError(predicted_labels, testlabels))
			print("Error on %dth fold testing set: %3.3f / 1.00"%(i+1, errors[-1]))
			# print("The output hypothesis:\n", wts)
		print("Mean Testing Error: %3.3f / 1.00"%(sum(errors)/k))
		print(wts)
	elif mode == "plot":
		plotErrorVsTGraph()
	elif mode == "genplot":
		generateDatapoints(dataset)
		plotErrorVsTGraph()

def ParseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action="store", dest="fin", default=None,
                     help='Specify dataset.')
  parser.add_argument('--mode', action="store", dest="mode", default="erm",
                     help='Specify mode,erm or kfold.')
  parser.add_argument('-k', type=int, dest="k", default=10,
  										help='Specify k for k fold cross validation.')
  parser.add_argument('-t', type=int, dest="t", default=5,
  										help='Specify the number of decision stumps for boosting.')
  return parser.parse_args()

def generateDatapoints(dataset):
	k = 10
	batches = getBatches(dataset, k)
	mean_val_error = []
	mean_train_error = []
	for T in range(1, 101):
		print(T)
		val_errors = []
		train_errors = []
		for i in range(k):
			# divide into testingset and training set
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
			
			classifiers = []
			# train adaboost
			wts = train(trainset, trainlabels, T, classifiers)
			predicted_train_labels = predict(trainset, classifiers)
			train_errors.append(calculateError(predicted_train_labels, trainlabels))
			
			# predict labels of testset and calculate error on test set
			predicted_val_labels = predict(testset, classifiers)
			val_errors.append(calculateError(predicted_val_labels, testlabels))
		
		mean_train_error.append(sum(train_errors)/k)
		mean_val_error.append(sum(val_errors)/k)
	print("TRAIN\n", mean_train_error, "VAL\n", mean_val_error )
	np.savetxt("plotpoints.csv", np.column_stack((mean_train_error, mean_val_error)), delimiter=",")

def plotErrorVsTGraph():
	plotpoints = np.genfromtxt("plotpoints.csv", delimiter=',')
	mean_train = plotpoints[:, 0]
	mean_val = plotpoints[:, 1]
	# print(mean_train, mean_val)
	plt.plot(mean_val, label="ER on Validation set")
	plt.plot(mean_train, label="ER on Trainig set")
	plt.legend()
	plt.xlabel('Value of T')
	plt.ylabel('Empirical Risk')
	plt.show()

if __name__ == "__main__":
	main()