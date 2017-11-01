from sklearn.cluster import KMeans
import numpy as np
from random import randint

def UniformSampling(dset):

	smallerdset = dset.get_datasets()['small']
	sizes = dset.get_datasets_sizes_ordered()
	(sizeLarger, sizeSmaller) = (sizes['big'], sizes['small'])

	ratio = int(sizeLarger/sizeSmaller) + 1
	delta = sizeLarger - sizeSmaller * ratio

	for key in ['X_train', 'y_train']:
		smallerdset[key] = np.repeat(smallerdset[key], ratio, axis=0)

		#remove rows if necessary
		if delta < 0:
			smallerdset[key] = smallerdset[key][0:delta]

def KMeansSampling(data):

	kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
	under_sampling = kmeans.cluster_centers_

def RandomSampling(dset):

	smallerdset = dset.get_datasets()['small']
	sizes = dset.get_datasets_sizes_ordered()

	(sizeLarger, sizeSmaller) = (sizes['big'], sizes['small'])

	print('tamanho: %d' %(sizeLarger - sizeSmaller))
	print('size_large: %d -- size_small: %d' %(sizes['big'], sizes['small']))

	for i in range(0, (sizeLarger - sizeSmaller)):
		index = randint(0, sizeSmaller)
		for key in ['X_train', 'y_train']:
			smallerdset[key] = np.append(smallerdset[key], [smallerdset[key][index]], axis=0)

	sizes = dset.get_datasets_sizes_ordered()
	print('size_large: %d -- size_small: %d' %(sizes['big'], sizes['small']))

def SMOTESampling(data):
	pass

