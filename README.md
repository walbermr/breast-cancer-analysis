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
We tried this arquicteture at first with no regularization. After that, we explored some regularizations to make the trainning more stable and trying get better results.

 * 0 regularization

 ![1L16N-RELUa](https://github.com/wmr1/breast-cancer-analysis/blob/master/images/1L16N-RELUa.png)

 Train Loss:       	0.0717 <br/>
 Validation Loss:  	0.1422 <br/>
 Test Loss: 		0.1346 <br/>
 Accuracy:         	0.9638 <br/>
 Recall:           	0.8281 <br/>
 Precision:        	0.4690 <br/>
 F1:               	0.5989 <br/>
 AUROC:            	0.9506 <br/>

 * 0.002 L2 regularization

2. 1L32N-RELU arquicteture (0.002 L2 regularization)
	

3. 1L32N-SIGMOID (0 regularization)


# References

[1] https://keras.io
[2] https://pandas.pydata.org
[3]	http://scikit-learn.org
[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.