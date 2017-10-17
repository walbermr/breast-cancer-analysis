from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

PATH = "./datasets/mammography-consolidated.csv"


def main():
	headers = ["f1", "f2","f3", "f4","f5", "f6","target"]
	dataset = pd.read_csv(PATH, names = headers)
	dataset.drop_duplicates(inplace = True)

	no_cancer = dataset[dataset["target"]==0]
	has_cancer = dataset[dataset["target"]==1]

	has_cancer = has_cancer.append([has_cancer]*29, ignore_index=True)
	has_cancer = has_cancer.drop(has_cancer.index[0:25])

	X_has_cancer = has_cancer.iloc[:, :-1].values
	y_has_cancer = has_cancer.iloc[:, -1].values
	## Treino: 50%, Validação: 25%, Teste: 25%
	X_has_cancer_train, X_has_cancer_test, y_has_cancer_train, y_has_cancer_test = \
	 train_test_split(X_has_cancer, y_has_cancer, test_size=1/4, random_state=42, stratify=y_has_cancer)

	X_has_cancer_train, X_has_cancer_val, y_has_cancer_train, y_has_cancer_val = \
	 train_test_split(X_has_cancer_train, y_has_cancer_train, test_size=1/3, random_state=42, stratify=y_has_cancer_train)


	X_no_cancer = has_cancer.iloc[:, :-1].values
	y_no_cancer = has_cancer.iloc[:, -1].values
	## Treino: 50%, Validação: 25%, Teste: 25%
	X_no_cancer_train, X_no_cancer_test, y_no_cancer_train, y_no_cancer_test = \
	 train_test_split(X_no_cancer, y_no_cancer, test_size=1/4, random_state=42, stratify=y_no_cancer)
	 
	X_no_cancer_train, X_has_cancer_val, y_no_cancer_train, y_no_cancer_val = \
	 train_test_split(X_no_cancer_train, y_no_cancer_train, test_size=1/3, random_state=42, stratify=y_no_cancer_train)


	return

if __name__ == "__main__":
	main()