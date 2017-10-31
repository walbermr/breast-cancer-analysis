from keras.models import Sequential
from keras.layers import Dense

class NeuralNetworkGenerator:

	architectures = []

	def __init__(self, path, epochs = 150, batch_size = 10):
		self.path = path
		self.epochs = epochs
		self.batch_size =  batch_size

		with open(path, "r") as f:
			qtd_archs = f.readline()
			for _ in range(0, int(qtd_archs)):
				activation = f.readline()
				layers = f.readline().split(" ")
				self.architectures.append({'activation' : activation.replace('\n', ''),
				                           'layers' : list(map(int, layers))
				                           })

	def evaluate(self, X, y):

		print("batch_size: %d epochs: %d" %(self.batch_size, self.epochs))

		for arch in self.architectures:

			# Create model.
			model = Sequential()

			for i in range(len(arch['layers'])):
				if i == 0:
					model.add(Dense(arch['layers'][i+1], input_dim = arch['layers'][i], activation = arch['activation']))
					i += 1
				elif i < len(arch['layers']) - 1:
					model.add(Dense(arch['layers'][i], activation = arch['activation']))
				else:
					model.add(Dense(arch['layers'][i], activation = 'sigmoid'))

			#Compile model.
			model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

			# Fit the model.
			model.fit(X, y, epochs = self.epochs, batch_size = self.batch_size)

			# Evaluate the model.
			scores = model.evaluate(X, y)
			print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
