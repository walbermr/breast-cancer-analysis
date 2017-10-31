import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetworkGenerator
from sklearn.model_selection import train_test_split

PATH = "./datasets/mammography-consolidated.csv"

class DataSet:
	def __init__(self, path):
		self.path = path

	def next_batch(self, size):
		pass 

def split_datatset(dataset):
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=1/4, random_state=42, stratify=y)

	X_train, X_val, y_train, y_val = \
		train_test_split(X_train, y_train, test_size=1/3, random_state=42, stratify=y_train)

	return {'X_train' : X_train, 
			'y_train' : y_train, 
			'X_test' : X_test, 
			'y_test' : y_test, 
			'X_val' : X_val, 
			'y_val' : y_val}

def UniformSampling(largerDataSet, smallerDataSet):

	sizeLarger = largerDataSet['X_train'].shape[0]
	sizeSmaller = smallerDataSet['X_train'].shape[0]

	ratio = int(sizeLarger/sizeSmaller) + 1
	delta = sizeLarger - sizeSmaller * ratio

	for key in smallerDataSet:
		smallerDataSet[key] = np.repeat(smallerDataSet[key], ratio, axis=0)

		#remove rows if necessary
		if(delta < 0):
			smallerDataSet[key] = smallerDataSet[key][0:delta]

def ConcatenateAndShuffleDataSet(ds1, ds2):

	X_train = np.concatenate((ds1['X_train'],ds2['X_train']), axis=0)
	y_train = np.concatenate((ds1['y_train'],ds2['y_train']), axis=0)
	X_test  = np.concatenate((ds1['X_test'], ds2['X_test']), axis=0)
	y_test  = np.concatenate((ds1['y_test'], ds2['y_test']), axis=0)
	X_val   = np.concatenate((ds1['X_val'], ds2['X_val']), axis=0)
	y_val   = np.concatenate((ds1['y_val'], ds2['y_val']), axis=0)		

	train = np.c_[X_train,y_train]
	val = np.c_[X_val,y_val]

	for _ in range(1,10):
		np.random.shuffle(train)
		np.random.shuffle(val)

	X_train = train[:, :-1]
	y_train = train[:, -1]

	X_val = val[:, :-1]
	y_val = val[:, -1]

	return [X_train, y_train, X_test, y_test, X_val, y_val]

def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = pd.read_csv(PATH, names = headers)
	dataset.drop_duplicates(inplace = True)

	noCancer  = dataset[dataset['target'] == 0]
	hasCancer = dataset[dataset['target'] == 1]

	# Split dataset 
	noCancerSplitted = split_datatset(noCancer)
	hasCancerSplitted = split_datatset(hasCancer)

	UniformSampling(noCancerSplitted, hasCancerSplitted)

	# Concatenating and Shuffling
	X_train, y_train, X_test, y_test, X_val, y_val = \
		ConcatenateAndShuffleDataSet(noCancerSplitted, hasCancerSplitted)

	# Build Model

	return


if __name__ == "__main__":

	nn = NeuralNetworkGenerator("nn.txt", epochs = 300)

#	main()