from sklearn.cluster import KMeans
import numpy as np

def KMeansSampling(data):

	kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
	under_sampling = kmeans.cluster_centers_

def RandomSampling(data):
	pass

def SMOTESampling(data):
	pass