import numpy as np
from numpy.lib.arraysetops import unique


### Assignment 3 ###

class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        #Lazy Algorithm, only stores data during training
        self.XT = X.copy()
        self.YT = y.copy()
        None

    def predict(self, X):
        # Run model here
        outputs = []

        # for each tuple in test dataset
        for x in X:
            closestKTuples = []
            euclideanDistances = []
            labels = self.YT.copy().tolist()

            # For each training tuple, calculate euclidean distance w.r.t. test data tuple
            for trainingXs in self.XT:
                euclideanDistances.append(self.distance(x, trainingXs))

            # retrieve the labels of the k nearest tuples/neighbours
            for i in range(self.k):
                closestKTuples.append(labels[euclideanDistances.index(min(euclideanDistances))])
                labels.pop(euclideanDistances.index(min(euclideanDistances)))
                euclideanDistances.remove(min(euclideanDistances))

            # append predicted label to output list by taking mode of the labels of closest K tuples
            outputs.append(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=closestKTuples))

        # return numpy array of predicted labels
        return np.asarray(outputs)


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels

        self.yT = np.where(y == 0, -1, y)
        self.XT = X.copy()

        # Weight update algorithm for specified number of steps
        for step in range(steps):
            index = step % len(self.XT)

            # Calculate label from the recently updated weights
            label = np.sign(np.sum(np.dot(self.w, self.XT[index])) + self.b)

            # If label doesn't match, update weights
            if label != self.yT[index]:
                self.w = self.w + self.lr * np.dot(self.yT[index], self.XT[index])
                self.b = self.b + self.lr * self.yT[index]

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        outputLabels = []
        for x in X:
            outputLabels.append(np.sign(np.dot(self.w, x) + self.b))

        outputLabels = np.array(outputLabels)
        return np.where(outputLabels == -1.0, 0, outputLabels).flatten()


class Node:
    def __init__(self) -> None:
        self.featureIndex = None
        self.featureValue = None
        self.label = None
        self.children = list()

    #for debug
    def __str__(self):
        # return f"{self.featureIndex} {self.featureValue} {self.label}"
        return f"indexValueToCheck: {self.featureIndex}: {[str(node) for node in self.children]} label: {self.label}"

class ID3:

    #modifying the np.unique method to return sorted list
    @staticmethod
    def unique(a):
        _, idx = unique(a, return_index=True)[1]
        return a[np.sort(idx)]

    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range
        self.root = None
        self.y = None

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    #Calculates the max entropy for current dataset
    def calcEntropy(self, feature):
        entropy = 0
        uniqueLabelCount = len(unique(feature))
        uniqueLabels = unique(feature)
        for i in range(uniqueLabelCount):
            probability = (np.count_nonzero(feature == uniqueLabels[i])) / len(feature)
            entropy = entropy - (probability * np.log2(probability))
        return entropy

    #Calculates Information for the features(Just the I value, without the multiplication)
    def calcI(self, outputLabels):
        I = 0
        sum = np.sum(outputLabels)
        for label in outputLabels:
            I = I - (label / sum) * np.log2(label / sum)
        return I

    #Calculates information needed by the feature to predict outcome
    def calcInformationNeeded(self, feature, y):
        outputLabels = []
        information = 0
        uniqueLabels = unique(feature)

        for uniqueLabel in uniqueLabels:
            dict = {}
            uniqueLabelIndices = np.where(feature == uniqueLabel)[0]
            for index in uniqueLabelIndices:
                if y[index] in dict:
                    dict[y[index]] = dict[y[index]] + 1
                else:
                    dict[y[index]] = 1
            for i in range(len(dict)):
                outputLabels = list(dict.values())
            I = self.calcI(outputLabels)

            occurance = (np.count_nonzero(feature == uniqueLabel))
            information = information + ((occurance / len(feature)) * (I))
        return information

    #Tree creation method
    def generateDecisionTree(self, X, y, parent):
        node = Node()

        #When dataset is empty
        if np.size(X) == 0:
            pluralityValue = max(set(parent), key=parent.count)
            node.label = pluralityValue
            node.featureIndex = None
            return node

        #When all the labels have same value
        if len(set(y)) == 1:
            node.label = y[0]
            node.featureIndex = None
            return node

        # calculate Initial Expected Information(entropy)
        entropy = self.calcEntropy(y)
        informationNeededList = []
        for i in range(len(X[0])):
            feature = X[:, i]
            informationNeededList.append(self.calcInformationNeeded(feature, y))

        #Calculate gain for all labels
        diff = entropy - np.array(informationNeededList)
        index = np.argmax(diff)

        #Select feature with max gain
        node = Node()
        node.featureIndex = index
        selectedFeature = X[:, index]
        X_playground = X.copy()
        y_playground = y.copy()
        labels = unique(selectedFeature)

        #if no attributes left
        if len(set(labels)) == 1:
            X = list(X[:, index])
            pluralityValue = max(set(X), key=X.count)
            node.label = pluralityValue
            return node

        #for each value in selected feature, create a subtree by recursively calling the tree generation method and add it into children list of current node
        for label in np.sort(labels):
            node.featureValue = label

            indices = np.where(X_playground[:, index] == label)
            X_playgroundForRowDelete = X_playground[indices]
            y_playgroundForRowDelete = y_playground[indices]

            # subtree = np.delete(X_playgroundForRowDelete, index, 1)
            node.children.append(self.generateDecisionTree(X_playgroundForRowDelete, y_playgroundForRowDelete, y))

        return node

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        categorical_data = self.preprocess(X)
        self.y = y.copy()
        self.root = self.generateDecisionTree(categorical_data, self.y, None)

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features

        outputLabels = []
        categorical_data = self.preprocess(X)

        for x in categorical_data:
            node = self.root

            while len(node.children) > 0:
                for child in node.children:
                    if child.featureValue is None or child.featureValue == x[node.featureIndex]:
                        node = child
                        break

            outputLabels.append(node.label)

        return np.array(outputLabels)


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
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

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
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # Write forward pass here
        return None

    def backward(self, gradients):
        # Write backward pass here
        return None


class Sigmoid:

    def __init__(self):
        None

    def forward(self, input):
        # Write forward pass here
        return None

    def backward(self, gradients):
        # Write backward pass here
        return None


class K_MEANS:

    def __init__(self, k, t):
        # k_means state here
        # Feel free to add methods
        # t is max number of iterations
        # k is the number of clusters
        self.k = k
        self.t = t

    def distance(self, centroids, datapoint):
        diffs = (centroids - datapoint) ** 2
        return np.sqrt(diffs.sum(axis=1))

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        return self.cluster
    # return array with cluster id corresponding to each item in dataset


class AGNES:
    # Use single link method(distance between cluster a and b = distance between closest
    # members of clusters a and b
    def __init__(self, k):
        # agnes state here
        # Feel free to add methods
        # k is the number of clusters
        self.k = k

    def distance(self, a, b):
        diffs = (a - b) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        return self.cluster
    # return array with cluster id corresponding to each item in dataset
