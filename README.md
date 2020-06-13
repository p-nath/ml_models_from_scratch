# How to run code
---

## 1. Perceptron
```
python3 perceptron.py -i <path_to_input_dataset> --epochs <number_of_epochs>  --mode <erm/k-fold> -k <value_of_k>
```

**mode:** used to choose between emperical risk minimization mode and k-fold cross validation mode; default is erm.

**k:** if k-fold mode is specified, use k flag to specify k; default is 10.

**i:** tested in linearly-separable-dataset.csv
---
## 2. Adaboost
```
python3 adaboost.py -i <path_to_input_dataset> --mode <erm/k-fold> -k <value_of_k> -t <number_of_stumps>
```
**mode:** used to choose between emperical risk minimization mode and k-fold cross validation mode; default is erm. 2 additional modes available - genplot to generate and store data to be plotted (Error v/s Number of Stumps(T)) into a file called "plotpoints.csv" and plot to plot the data stored in "plotpoints.csv".

**k:** if k-fold mode is specified, use k flag to specify k; default is 10.

**i:** tested in Breast_cancer_data.csv

**t:** number of decision stumps you want to use for boosting

---
<br>

## 3. Support Vector Machine

Implemented Soft-SVM Using SGD Algortihm from Understanding Machine Learning: From Theory to Algorithms(15.5). Tested on a linearly separable dataset with two clusters. Since it is linearly separable, the model showed an accuracy of 100 percent.

---
<br>

## 4. K-means Clustering
```
python3 kmeans.py -i <path_to_input_file> --distance_metric <euclidean/manhattan> -k <k_value> --num_epochs <num_of_epochs> --plot --scale
```
Note
- distance_metric is set to euclidean by default
The following are the optional arguments
- k value is set to 2 by default
- num_epochs is set to 5 by default
- plot is set to false by default
- scale is set to false by default (scaling the data gives better positive diagnosis)


**To run script using euclidean metric**
```
python3 kmeans.py -i Breast_cancer_data.csv --distance_metric manhattan --scale
```

**To run script using manhattan metric**
```
python3 kmeans.py -i Breast_cancer_data.csv --distance_metric euclidean --scale
```


**How positive diagnoses are calculated**
For every pair of known cluster and predicted cluster, count the number of datapoints correctly separated. Return count divided by the size of the known cluster.


**Implementation**
1. initialze k centres at random
2. run till number of epochs have been completed or algorithm converges
	- find the distances between a each datapoint and the centres and assign the datapoints to its closest centre
	- shift the centres based on the arithmetic mean of the datapoints assigned to tha cluster
	
---
<br>

## 5. k-Nearest Neighbours
```
python3 knn.py -i <path_to_input_file> -k <number_of_neighbours>
```
Notes
- tested in Breast_cancer_data.csv
- path_to_input_file needs to be mentioned everytime while running the script
- default value of k is 10 in case it is not mentioned
- the script does not require a seed but it can be mentioned if desired by appending '--seed <seed_val>'

**Testing**

The dataset is shuffled randomly (with a seed if you desire) and split into a training dataset(80%) and a testing dataset(20%).
The following table shows how the model performed on the dataset using different values of k.
```
| Value of k | Accuracy on 1st Run | Accuracy on 2nd Run |
  ----------   -------------------   -------------------
1.    2				85.97 %			86.09 %
2.	4				91.23 %			90.35 %
3.	6				84.21 %			87.72 %
4.	8				94.74 %			90.36 %
5.    10			   91.22 %			88.60 %

```

**Implementation**

My implementation uses euclidian distance to perform k-nearest neighbours clustering. 

1. Load breast cancer data and split it into testing data and training data in a 20: 80 ratio.
2. Iterate throught the test dataset and call predictKNN( ) to get the predicted label.
3. predictKNN( ) 
	- computes the distance between each training datapoint and the test datapoint
	- get the closest k training datapoints and return the most commonly occuring label as the label of the test datapoint
4. Compare with labels of the testset and print accuracy in percentage.

---

NB: All the algorithms have been implemented using *Understanding Machine Learning: From Theory to Algorithms* as a reference.