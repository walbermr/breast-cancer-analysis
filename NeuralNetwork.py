from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from Util import plot_training_error_curves
from keras.regularizers import l2
import numpy as np

class NeuralNetworkGenerator:

	architectures = []
	architectures_scores = []

	def __init__(self, path, epochs = 150, batch_size = 50):
		self.path = path
		self.epochs = epochs
		self.batch_size =  batch_size

		with open(path, "r") as f:
			qtd_archs = f.readline()
			for _ in range(0, int(qtd_archs)):
				activation, regularization = f.readline().split(" ")
				layers = f.readline().split(" ")
				self.architectures.append({'regularization' : float(regularization),
										   'activation' : activation.replace('\n', ''),
				                           'layers' : list(map(int, layers))
				                           })

	def evaluate(self, dataset):
		print("Evaluating achitectures: \n")
		print("batch_size: %d epochs: %d" %(self.batch_size, self.epochs))

		np.random.seed(7)
		x = 1
		for arch in self.architectures:
			# Create model.
			model = Sequential()

			for i in range(len(arch['layers'])):
				if i == 0:
					model.add(Dense(arch['layers'][i+1], input_dim = arch['layers'][i], activation = arch['activation']))
					i += 1
				elif i < len(arch['layers']) - 1:
					
					# if arch['regularization'] > 0:
						#model.add(Dropout(0.2))
						model.add(Dense(arch['layers'][i], kernel_initializer='random_normal',  activation = arch['activation'], 
									activity_regularizer=l2(arch['regularization'])))
					# else:
					# 	model.add(Dense(arch['layers'][i], activation = arch['activation']))

				else:
					#model.add(Dropout(0.2))
					model.add(Dense(arch['layers'][i], activation = 'sigmoid'))

			# Compile model.
			model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

			# Get dataset variables
			dset = dataset.get_dataset()

			# Fit the model.
			history = model.fit(dset['X_train'], dset['y_train'], epochs = self.epochs, batch_size = self.batch_size,
			                    callbacks = [EarlyStopping(monitor='val_loss', patience = 10)], validation_data = (dset['X_val'], dset['y_val']))

			# Evaluate the model.

			scores = model.evaluate(dset['X_test'], dset['y_test'])
			self.architectures_scores.append((scores, len(history.history['loss'])))
			plot_training_error_curves(history, x)

			print("Architecture %d: " %x)
			print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
			print("--------------------------------------------------------------")
			x += 1

		##self.store_test_scores()

	def store_test_scores(self):
		print("Storing test scores...")

		i = 1
		with open("results/output.txt", "w") as f:
			for scores in self.architectures_scores:
				f.write("\nArchitecture %d \n%s: %.2f%% (%d)" % (i, "accuracy", scores[0][1] * 100, scores[1]))
				i += 1
