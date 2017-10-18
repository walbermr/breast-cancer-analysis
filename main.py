import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = "./datasets/mammography-consolidated.csv"

def split_datatset(dataset):
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values

	X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=1/4, random_state=42, stratify=y)

	X_train, X_val, y_train, y_val = \
		train_test_split(X_train, y_train, test_size=1/3, random_state=42, stratify=y_train)

	return {'X_train' : X_train, 
			'y_train' : y_train, 
			'X_test' : X_test, 
			'y_test' : y_test, 
			'X_val' : X_val, 
			'y_val' : y_val}

#def 

def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = pd.read_csv(PATH, names = headers)
	dataset.drop_duplicates(inplace = True)

	no_cancer = dataset[dataset["target"]==0]
	has_cancer = dataset[dataset["target"]==1]

	# ToDO: make it more generic
	has_cancer = has_cancer.append([has_cancer]*29, ignore_index=True)
	has_cancer = has_cancer.drop(has_cancer.index[0:25])

	# Split dataset 
	no_cancer_splitted = split_datatset(no_cancer)
	has_cancer_splitted = split_datatset(has_cancer)

	# merge dataframes

	return

if __name__ == "__main__":
	main()