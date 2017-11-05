import numpy as np
import matplotlib.pyplot as plt

def extract_final_losses(history):
	"""Função para extrair o melhor loss de treino e validação.

	Argumento(s):
	history -- Objeto retornado pela função fit do keras.

	Retorno:
	Dicionário contendo o melhor loss de treino e de validação baseado
	no menor loss de validação.
	"""
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	idx_min_val_loss = np.argmin(val_loss)
	return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history, arch_idx = None):
	"""Função para plotar as curvas de erro do treinamento da rede neural.

	Argumento(s):
	history -- Objeto retornado pela função fit do keras.

	Retorno:
	A função gera o gráfico do treino da rede e retorna None.
	"""
	print("Ploting training error curves...")

	train_loss = history.history['loss']
	val_loss = history.history['val_loss']

	fig, ax = plt.subplots()
	ax.plot(train_loss, label = 'Train')
	ax.plot(val_loss, label = 'Validation')
	ax.set(title = 'Training and Validation Error Curves', xlabel = 'Epochs', ylabel = 'Loss (MSE)')
	ax.legend()

	#plt.show()
	file = "plot.png"
	if arch_idx:
		file = "arch_" + str(arch_idx) + "_" + file

	file = "results/" + file

	fig.savefig(file)