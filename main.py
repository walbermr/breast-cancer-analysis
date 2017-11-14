from NeuralNetwork import NeuralNetworkGenerator
from SamplingFunctions import *
from datasetManager.dataset import DataSet

def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = DataSet("./datasets/mammography-consolidated.csv", headers, UniformSampling)

	# Build Model
	nn = NeuralNetworkGenerator("nn.txt", epochs = 300)
	nn.evaluate(dataset)

if __name__ == "__main__":
	main()