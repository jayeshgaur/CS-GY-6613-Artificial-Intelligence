import numpy
import numpy as np
import math


### Assignment 4 ###

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b
		# We need input during backpropagation
		self.x = None

	def forward(self, input):
		self.x = input
		return (np.matmul(input, self.w) + self.b)

	def backward(self, gradients):
		wDash = np.matmul(np.transpose(self.x), gradients)
		updatedX = np.matmul(gradients, np.transpose(self.w))
		self.w = self.w - self.lr * wDash
		self.b = self.b - self.lr * gradients
		return updatedX


class Sigmoid:

	def __init__(self):
		# Needed for backpropagation
		self.sigmoidValues = None

	def forward(self, input):
		neuronOutput = 1 / (1 + np.exp(-(input)))
		self.sigmoidValues = neuronOutput
		return self.sigmoidValues

	def backward(self, gradients):
		return gradients * (1 - self.sigmoidValues) * (
				(self.sigmoidValues))


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t
		self.cluster = None

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum())
		# return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		# input is array of features (no labels)

		# Copying the data for processing
		self.cluster = X.copy()

		# Adding one more column to store cluster number for each data tuple
		clusterColumn = np.arange(np.shape(X)[0])
		self.cluster = np.column_stack((self.cluster, clusterColumn))

		# Select initial k centroids randomly
		random_centroid_indexes = np.random.choice(np.shape(X)[0], size=self.k, replace=False)
		centroids = X[random_centroid_indexes, :]

		# Initialize a array with column length = k, to store distance of each data point with each centroid
		distanceArray = np.arange(self.k * np.shape(X)[0]).reshape(np.shape(X)[0], self.k)

		for t in range(self.t):

			# Calculate distance of each data point from each centroid
			for xIndex in range(np.shape(X)[0]):
				for index in range(self.k):
					distanceArray[xIndex][index] = self.distance(centroids[index], X[xIndex])

			clusterColumnIndex = self.cluster.shape[1] - 1

			# Find minimum distance and update cluster number in the last column
			for rowIndex in range(np.shape(X)[0]):
				clusterIndex = np.argmin(distanceArray[rowIndex])
				self.cluster[rowIndex][clusterColumnIndex] = clusterIndex

			# Find unique cluster labels
			clusters = np.unique(self.cluster[:,clusterColumnIndex])

			# Backup old centroids
			old_centroids = centroids.copy()
			centroids = None

			# For each cluster, evaluate a new centroid.
			for cluster in clusters:
				i = 0
				sum = np.zeros(clusterColumnIndex)
				for index in range(np.shape(X)[0]):
					if self.cluster[index][clusterColumnIndex] == cluster:
						sum = np.add(X[index],sum)
						i = i + 1
				sum = sum / i

				# Save the new centroid in centroids
				centroids = sum if centroids is None else np.vstack((centroids, sum))

				# If there is no change in centroids, terminate the algorithm and return the current cluster labels
				if np.array_equal(old_centroids,centroids):
					return self.cluster[:, clusterColumnIndex]

		return self.cluster[:,clusterColumnIndex]
		#return array with cluster id corresponding to each item in dataset

class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):

		#training logic here
		#input is array of features (no labels)

		# Create a 2d matrix to save distances of all points
		distanceMatrix = np.zeros(((np.shape(X)[0]),(np.shape(X)[0])))

		# Calculate distance of each tuple from each other
		# We use upper triangular matrix here to prevent needless calculations
		for i in range(np.shape(X)[0]):
			for j in range(np.shape(X)[0]):
				if i < j:
					distance = self.distance(X[i], X[j])
					distanceMatrix[i][j] = distance
				else:
					distanceMatrix[i][j] = np.inf

		# Initialize clusters. At first, each tuple is a unique cluster.
		# We use two data structures to maintain our cluster organisation

		# Dictionary to maintain cluster -> data points
		# key corresponds to cluster and values correspond to data points
		cluster_dictionary = {}
		for i in range(np.shape(X)[0]):
			tupleList = list(range(i,i+1))
			cluster_dictionary.update({i: tupleList})

		# List to maintain data point -> cluster
		# index corresponds to data point, value corresponds to cluster
		cluster_list = list(range(0,len(X)))

		# Club the least dissimilar tuples together until we have k clusters
		while len(cluster_dictionary.keys()) > self.k:

			# Get the indices of the two tuples with the least distance
			i,j = (np.unravel_index(distanceMatrix.argmin(), distanceMatrix.shape))

			# Change distance to infinity to not run into the same pair again
			distanceMatrix[i][j] = np.inf

			# Get current clusters of i and j
			cluster_i = cluster_list[i]
			cluster_j = cluster_list[j]

			# If the clusters are different, combine them all into cluster i
			if cluster_j != cluster_i:
				# Retrieve indices of tuples in cluster j
				cluster_j_tuples = cluster_dictionary.get(cluster_j)

				# Update the cluster of the above retrieved tuples to i
				for tuple in cluster_j_tuples:
					cluster_list[tuple] = cluster_i

				# Retrieve indices of tuples in cluster i
				cluster_i_tuples = cluster_dictionary.get(cluster_i)

				# Combine tuples of both clusters
				combined_new_tuples = cluster_i_tuples + cluster_j_tuples

				# Tuples already in the same cluster do not need to be reevaluated against each other
				# Therefore, discard their distances with each other
				for i in combined_new_tuples:
					for j in combined_new_tuples:
						distanceMatrix[i][j] = np.inf

				# Delete cluster j since we merged everything into cluster i
				cluster_dictionary.pop(cluster_j)

				# Update the tuples of cluster i in dictionary
				cluster_dictionary.update({cluster_i: combined_new_tuples})

		return cluster_list
