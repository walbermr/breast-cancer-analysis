import numpy as np
import pandas as pd
from datasetManager.dataset import DataSet
from NeuralNetwork import  *
from sklearn.model_selection import train_test_split

PATH = "./datasets/mammography-consolidated.csv"


def UniformSampling(dset):

	sizes = dset.get_datasets_sizes()

	smallerdset = dset.get_datasets()['small']

	(sizeLarger, sizeSmaller) = (sizes['hasCancer'], sizes['noCancer']) \
									if (sizes['hasCancer'] > sizes['noCancer']) else \
								(sizes['noCancer'],sizes['hasCancer'])

	ratio = int(sizeLarger/sizeSmaller) + 1
	delta = sizeLarger - sizeSmaller * ratio

	for key in smallerdset:
		smallerdset[key] = np.repeat(smallerdset[key], ratio, axis=0)

		#remove rows if necessary
		if(delta < 0):
			smallerdset[key] = smallerdset[key][0:delta]

def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = DataSet(PATH, headers)

	UniformSampling(dataset)

	# Concatenating and Shuffling
	dataset.ConcatenateAndShuffleDataSet()

	# Build Model

	return


if __name__ == "__main__":

	nn = NeuralNetworkGenerator("nn.txt", batch_size = 30)

	nn.evaluate([], [])

#	main()