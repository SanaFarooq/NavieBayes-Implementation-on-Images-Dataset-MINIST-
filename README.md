# NavieBayes-Implementation-on-Images-Dataset-MINIST-
****1: Describe Naïve Bayes
**Naïve Bayes is a probabilistic machine learning algorithm based on Bayes’ theorem. It is commonly used for classification, particularly in NLP and image recognition.
Discrete Naïve Bayes that feature a categorical or discrete structure. For example, the count of specific words in a text document is discrete in nature.
Steps involved in Discrete Naïve Bayes:
Calculate probability of each class in the dataset. 
(Formula:  P(Class = c) = Count of instances of class c / Total number of instances)
Calculate probability of a feature in a given class.
(Formula: P(Feature| Class = c) = Count of instances with features in c / Total count of instances in c )
Given a new instance with features, compute the probability of each class using Bayes' theorem.
(Formula: P(Class = c ∣ features) = P(Class = c) × ∏ni=1   ​P (feature ​∣ Class = c) )

Continuous Naïve Bayes assumes continuous or real valued features. It models features using continuous probability distributions like Gaussian. Sensor readings are examples of continuous features.
Steps involved in Continuous Naïve Bayes:
Calculate probability of each class in the dataset. 
(Formula:  P(Class = c) = Count of instances of class c / Total number of instances)
Conditional Probability (P(feature | class)):
Estimate parameters of a probability distribution (usually Gaussian) for each feature within each class (mean and variance).
For a feature x in class c:
Mean = (i=1Ncxi )/Nc   , where  Nc is the number of instances in class c
Variance 2 c,x=(i=1Nc(xi - c,  x))/Nc 
Prediction: Similar to discrete Naïve Bayes but using probability density functions for continuous features.

Classification using Naïve Bayes has a training phase and a prediction phase.
Training Phase:
Discrete Case:
Calculate the prior probabilities for each class (P(class)).
Estimate conditional probabilities for each feature given each class (P(feature|class)).
Continuous Case:
Compute mean and variance for each feature within each class.

Prediction Phase:
Given new data (set of features), compute the likelihood of these features for each class.
Use Bayes' theorem to calculate the posterior probability for each class.
Assign the class with the highest posterior probability as the prediction.

Estimating Conditional Probability Distributions:
Discrete Case:
Compute probabilities using relative frequency counts.
For instance, for a particular feature given a class, count the occurrences of that feature in that class and divide by the total count of that class.
Continuous Case:
For Gaussian distribution, estimate the mean and variance for each feature within each class.
Use the sample mean and variance of the feature values observed within each class.
Smoothing in Naïve Bayes:
Why Smoothing?
To handle scenarios where certain feature class combinations were not present in the training data.
Avoid zero probabilities which could lead to biased predictions.
Smoothing Techniques:
Discrete Case (Additive or Laplace Smoothing):
Add a small value to all counts (usually 1) to avoid zero probabilities.
Continuous Case:
Add a small value (epsilon) to the variance to prevent probabilities from becoming zero.
Implementing Naïve Bayes involves computing these probabilities and applying the appropriate techniques based on the nature of the features (discrete or continuous) and handling potential issues like zero probabilities through smoothing.



Results Analysis:

Results Analysis:
Accuracy:
Test Set Accuracy: 55.58% (from the test confusion matrix)
Training Set Accuracy: 9.26% (from the training confusion matrix)
Precision, Recall, F1score:
Vary significantly across different classes.
The classifier performs relatively well on certain classes (e.g., 1, 7) and poorly on others (e.g., 5, 8).

fig 1: precision recall and F1 score at test Data

FIG 2 : precision recall and F1 score at test Data

Challenges Faced:
Handling High Dimensional Data: Naïve Bayes assumes feature independence, which might not hold well in high dimensional data like images.
Distribution Assumptions: Gaussian assumptions for continuous features might not perfectly match the true distribution of pixel values.
Class Imbalance: Some digits might be underrepresented, affecting model performance.
Possible Improvements:
Feature Engineering: Extract more informative features from images that might better represent the digits.
Model Selection: Explore other classification models that can handle high dimensional data better than Naïve Bayes, like ensemble methods or deep learning models.
Normalization or Transformation: Preprocess image data to ensure better conformance with assumptions made by Naïve Bayes or other models.
Address Class Imbalance: Strategies like oversampling or under sampling the minority/majority classes could help balance the dataset.
Further Investigation:
Analyze misclassified instances or classes to understand why certain digits are harder to predict.
Experiment with different hyperparameters or preprocessing techniques to enhance model performance.

Improving the Naïve Bayes classifier's performance might involve a combination of data preprocessing, model selection, and hyperparameter tuning to better suit the complexities of the MNIST dataset.


Documentation of Code Snaps:


Libraries Importing:
The libraries imported in the code snippet you provided are used for implementing a Naive Bayes classifier on the MNIST dataset. Here's a brief description of each library:
NumPy (import numpy as np):
NumPy is a powerful numerical computing library in Python.
It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
In machine learning, NumPy is commonly used for handling data in the form of arrays.
Matplotlib (import matplotlib.pyplot as plt):
Matplotlib is a 2D plotting library for creating static, animated, and interactive visualizations in Python.
The pyplot module provides a convenient interface for creating various types of plots and charts.
In this code, it might be used for visualizing data or results, although specific usage is not evident from the provided snippet.
Scikit-learn's Gaussian Naive Bayes (from sklearn.naive_bayes import GaussianNB):
Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data analysis and modeling.
Gaussian Naive Bayes is a variant of the Naive Bayes algorithm, specifically designed for continuous data assuming a Gaussian distribution.
It is commonly used for classification tasks.
Scikit-learn's Confusion Matrix (from sklearn.metrics import confusion_matrix):
A confusion matrix is a table used to evaluate the performance of a classification algorithm.
It shows the counts of true positive, true negative, false positive, and false negative predictions.
The confusion_matrix function from scikit-learn is used to compute the confusion matrix for a set of predictions.
TensorFlow's Keras (from tensorflow.keras.datasets import mnist):
TensorFlow is an open-source machine learning library developed by the Google Brain team.
Keras is a high-level neural networks API that runs on top of TensorFlow.
The mnist module from tensorflow.keras.datasets provides access to the MNIST dataset, a widely used dataset for handwritten digit recognition.
In summary, these libraries are utilized for loading the MNIST dataset, implementing a Gaussian Naive Bayes classifier, and evaluating its performance using a confusion matrix. Matplotlib may be used for optional visualization aspects in the code.

Explanation of each Line of Code:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
import numpy as np: Imports the NumPy library and aliases it as np for convenience. NumPy is used for numerical operations in Python.
import matplotlib.pyplot as plt: Imports the pyplot module from the Matplotlib library and aliases it as plt. Matplotlib is a plotting library used for creating visualizations.
from sklearn.naive_bayes import GaussianNB: Imports the Gaussian Naive Bayes classifier from scikit-learn. This classifier is suitable for datasets with continuous features assumed to be normally distributed.
from sklearn.metrics import confusion_matrix: Imports the confusion_matrix function from scikit-learn, which is used to evaluate the performance of a classification algorithm.
from tensorflow.keras.datasets import mnist: Imports the MNIST dataset from TensorFlow's Keras API. This dataset contains images of handwritten digits.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data(): Loads the MNIST dataset into four variables: x_train and y_train for training data, and x_test and y_test for test data.

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.reshape(60000, 784): Reshapes the training data to a 2D array with 60000 rows and 784 columns (28x28 pixels flattened).
x_test = x_test.reshape(10000, 784): Reshapes the test data similarly.
plt.imshow(x_train[8].reshape((28, 28)), cmap='gray')
plt.show()

plt.imshow(x_train[8].reshape((28, 28)), cmap='gray'): Displays the 8th training image using Matplotlib. The cmap='gray' argument specifies a grayscale colormap.
nb_model = GaussianNB()
fit_nb = nb_model.fit(x_train, y_train)

nb_model = GaussianNB(): Initializes a Gaussian Naive Bayes model.
fit_nb = nb_model.fit(x_train, y_train): Fits the Naive Bayes model to the training data.
predictions = fit_nb.predict(x_test)

predictions = fit_nb.predict(x_test): Uses the trained model to make predictions on the test data.
con_matrix = confusion_matrix(y_test, predictions)
print(con_matrix)

con_matrix = confusion_matrix(y_test, predictions): Computes the confusion matrix using the true labels (y_test) and the predicted labels (predictions) on the test set.
def diagonal_sum(con_matrix):
    sum = 0
    for i in range(10):
        for j in range(10):
            if i == j:
                sum += con_matrix[i, j]

    return sum

sum = diagonal_sum(con_matrix)
print(sum)
print(f'Accuracy % : {sum/10000}')

Defines a function diagonal_sum to calculate the sum of diagonal elements in the confusion matrix.
Computes the sum of diagonal elements and calculates accuracy as the ratio of the sum to the total number of samples.

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

Prints a classification report, which includes precision, recall, and F1-score for each class in the test set

The same process goes for test data as well.

**Summary**:
In summary, this code loads the MNIST dataset, applies a Gaussian Naive Bayes classifier, evaluates its performance using confusion matrices and classification reports, and visualizes the confusion matrices. The final lines print the accuracy for both the training and test sets
For more updates or any other questions you can ping me.
