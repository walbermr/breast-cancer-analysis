from sklearn.cluster import KMeans
from datasetManager.dataframes import *
import numpy as np
from random import randint

def UniformSampling(df1, df2):

	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']
	
	(sizelarger, sizesmaller) = get_dataframes_sizes(biggerdset, smallerdset)

	ratio = int(sizelarger/sizesmaller) + 1
	delta = sizelarger - sizesmaller * ratio

	for key in ['X_train', 'y_train']:
		smallerdset[key] = np.repeat(smallerdset[key], ratio, axis=0)

		#remove rows if necessary
		if delta < 0:
			smallerdset[key] = smallerdset[key][0:delta]

	return (smallerdset, biggerdset)

def KMeansSampling(dset):

	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']

	kmeans = KMeans(n_clusters = 2, random_state = 0).fit(biggerdset["X_train"])
	under_sampling = kmeans.cluster_centers_

	return (smallerdset, under_sampling)


def RandomSampling(df1, df2):

	smallerdset = get_ordered_dataframes(df1, df2)['small']
	biggerdset = get_ordered_dataframes(df1, df2)['big']

	(sizelarger, sizesmaller) = get_dataframes_sizes(biggerdset, smallerdset)

	for i in range(0, (sizelarger - sizesmaller)):
		index = randint(0, sizesmaller)
		for key in ['X_train', 'y_train']:
			smallerdset[key] = np.append(smallerdset[key], [smallerdset[key][index]], axis=0)

	return (smallerdset, biggerdset)

def SMOTESampling(dset):
	pass

