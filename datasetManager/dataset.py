import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet:
	def __init__(self, path, headers):
		self.path = path
		self.headers = headers
		self.dataframes = {}
		self.spldataframes = {}

		self.dataframes['main'] = pd.read_csv(self.path, names = headers)

		self.sampling_funcs[sampling]('noCancerSplitted', 'hasCancerSplitted')


	def next_batch(self, size):
		pass

	def __create_spl_dframe(a,b,c,d,e,f):

		return {'X_train' : a, 
				'y_train' : b, 
				'X_test' : c, 
				'y_test' : d, 
				'X_val' : e, 
				'y_val' : f}

	def select_target(self, dframe, feat, value):

		return self.dataframes[dframe][self.dataframes[dframe][feat] == value]

	def split_dataframe(self, dframe):
		X = self.dataset.dataframes[dframe].iloc[:, :-1].values
		y = self.dataset.dataframes[dframe].iloc[:, -1].values

		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size=1/4, random_state=42, stratify=y)

		X_train, X_val, y_train, y_val = \
			train_test_split(X_train, y_train, test_size=1/3, random_state=42, stratify=y_train)

		return __create_spl_dframe(X_train, y_train, X_test, y_test, X_val, y_val)

	def ConcatenateAndShuffleDataSet(self, ds1, ds2):

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

		return __create_spl_dframe(X_train, y_train, X_test, y_test, X_val, y_val)
