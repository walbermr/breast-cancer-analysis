# Breast Cancer Analysis

We used machine learning algorithms to build a classification model which learns carcinogenic individual's patterns and predicts if a new individual has or no cancer. The analyzed dataset contains about 2000 lines with 6 different features extracted from breast image exams and the target which indicates if the individual was diagnosed with cancer or not. This project was developed in Python using Keras [1], Pandas [2] and scikit-learn [3] libraries.

# Running

Follow the steps:

	$ conda env create -f environment.yml
	$ activate breast_cancer_analysis
	$ python main.py

# Features
* Neural network implementation using Keras;
* Many different sampling functions (uniform sampling, SMOTE sampling, and Random Sampling);
* Uses different metrics to evaluate the results (AUROC, Recall, Precision, F1 and accuracy).

# Results

### Neural Networks ###

We tested several different architectures (we changed number of hidden layers, number of neurons, activation function, optimizer, and regularization), those which presented the bests results were chosen trying to get better results after different improvements. Most of tested architectures were ommited here in order to keep this report brief. 

1. 1L16N-RELU arquicteture
This arquicteture consists of one hidden layer with 16 neurons and ReLU[4] activation function.
We tried this arquicteture at firt with no regularization. After that, we explored some regularizations to make the trainning more stable and trying get better results.

* 

*

2.  


3. 


# References

[1] https://keras.io
[2] https://pandas.pydata.org
[3]	http://scikit-learn.org
[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.