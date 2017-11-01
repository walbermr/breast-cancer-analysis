from sklearn.cluster import KMeans
import numpy as np

def UniformSampling(dset):

	smallerdset = dset.get_datasets()['small']
	sizes = dset.get_datasets_sizes_ordered()
	(sizeLarger, sizeSmaller) = (sizes['big'], sizes['small'])

	ratio = int(sizeLarger/sizeSmaller) + 1
	delta = sizeLarger - sizeSmaller * ratio

	for key in smallerdset:
		smallerdset[key] = np.repeat(smallerdset[key], ratio, axis=0)

		#remove rows if necessary
		if delta < 0:
			smallerdset[key] = smallerdset[key][0:delta]

def KMeansSampling(dset):
	n_clusters = dset.get_datasets_sizes_ordered()['small']
	bigdset = dset.get_datasets()['big']

	kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(bigdset["X_train"])
	Xs = kmeans.cluster_centers_

	#dset.set(X = Xs)


def RandomSampling(dset):

	smallerdset = dset.get_datasets()['small']
	small_size = dset.get_datasets_sizes()

def SMOTESampling(dset):
	pass

