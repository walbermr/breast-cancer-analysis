import numpy as np
import pandas as pd
from datasetManager.dataset import DataSet
from NeuralNetwork import  *
from sklearn.model_selection import train_test_split

PATH = "./datasets/mammography-consolidated.csv"


def UniformSampling(dset, largedset, smallerdset):

	sizeLarger = dset.spldataframes[largedset]['X_train'].shape[0]
	sizeSmaller = dset.spldataframes[smallerdset]['X_train'].shape[0]

	ratio = int(sizeLarger/sizeSmaller) + 1
	delta = sizeLarger - sizeSmaller * ratio

	for key in dset.spldataframes[smallerdset]:
		dset.spldataframes[smallerdset][key] = \
			np.repeat(dset.spldataframes[smallerdset][key], ratio, axis=0)

		#remove rows if necessary
		if(delta < 0):
			dset.spldataframes[smallerdset][key] = \
				dset.spldataframes[smallerdset][key][0:delta]

def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = DataSet(PATH, headers)

	dataset.dataframe.drop_duplicates(inplace = True)

	dataset.dataframes['noCancer'] = dataset.select_target('main', 'target', 0)
	dataset.dataframes['hasCancer'] = dataset.select_target('main', 'target', 1)

	# Split dataset 
	dataset.spldataframes['noCancerSplitted'] = dataset.split_dataframe('noCancer')
	dataset.spldataframes['hasCancerSplitted'] = dataset.split_dataframe('hasCancer')

	UniformSampling(dataset, 'noCancerSplitted', 'hasCancerSplitted')


	# Concatenating and Shuffling
	X_train, y_train, X_test, y_test, X_val, y_val = \
		ConcatenateAndShuffleDataSet(noCancerSplitted, hasCancerSplitted)

	# Build Model

	return


if __name__ == "__main__":

	nn = NeuralNetworkGenerator("nn.txt", batch_size = 30)

	nn.evaluate([], [])

#	main()