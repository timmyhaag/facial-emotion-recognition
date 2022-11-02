'''
Data set introduction
The data consists of 48x48 pixel grayscale images of faces
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The faces have been automatically registered so that the face is more or less centered
and occupies about the same amount of space in each image
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


''' ### Read csv data '''
df = pd.read_csv('train.csv')
print("There are total ", len(df), " sample in the loaded dataset.")
print("The size of the dataset is: ", df.shape)
# get a subset of the whole data for now
df = df.sample(frac=0.1, random_state=46)
print("The size of the dataset is: ", df.shape)

''' Extract images and label from the dataframe df '''
width, height = 48, 48
images = df['pixels'].tolist()
faces = []
for sample in images:
    face = [int(pixel) for pixel in sample.split(' ')]  # Splitting the string by space character as a list
    face = np.asarray(face).reshape(width * height)  # convert pixels to images and # Resizing the image
    faces.append(face.astype('float32') / 255.0)  # Normalization
faces = np.asarray(faces)

# Get labels
y = df['emotion'].values

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Visualization a few sample images
plt.figure(figsize=(5, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(faces[i].reshape(width, height)), cmap='gray')
    plt.xlabel(class_names[y[i]])
plt.show()

## Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces, y, test_size=0.40, random_state=46)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.svm import SVC
from sklearn import svm

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# Now that our classifier has been trained, let's make predictions on the test data. To make predictions, the predict method of the DecisionTreeClassifier class is used.
y_pred = svclassifier.predict(X_test)

# For classification tasks some commonly used metrics are confusion matrix, precision, recall, and F1 score.
# These are calculated by using sklearn's metrics library contains the classification_report and confusion_matrix methods
from sklearn.metrics import classification_report, confusion_matrix
print("\nPerformance results for a Support Vector Machine with a linear kernel:\n")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Train SVM with rbf kernel and make predictions on test data
svclassifier_rbf = svm.SVC(kernel='rbf')
svclassifier_rbf.fit(X_train, y_train)
y_pred2 = svclassifier_rbf.predict(X_test)
print("\nPerformance results for a Support Vector Machine with an RBF kernel:\n")
print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))


# Train decision model and make predictions on test data
decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
decision_tree.fit(X_train, y_train)
y_pred3 = decision_tree.predict(X_test)
print("\nPerformance results for an ID3 decision tree:\n")
print(confusion_matrix(y_test, y_pred3))
print(classification_report(y_test, y_pred3))


# Train naive bayes model and make predictions on test data
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)
y_pred4 = bayes_classifier.predict(X_test)
print("\nPerformance results for a Naïve Bayes algorithm:\n")
print(confusion_matrix(y_test, y_pred4))
print(classification_report(y_test, y_pred4))

# Calculate total leaf nodes and nodes in decision tree and output results

tree_node_count = decision_tree.tree_
print("\nTotal number of leaf nodes in the decision tree: ", decision_tree.get_n_leaves())
print("Total number of nodes in the decision tree: ", tree_node_count.node_count)

# Display performance accuracies for each implemented machine learning algorithm.

print("\nAccuracy for SVM linear kernel:", metrics.accuracy_score(y_test, y_pred))
print("Accuracy for SVM rbf kernel:", metrics.accuracy_score(y_test, y_pred2))
print("Accuracy for ID3 decision tree:", metrics.accuracy_score(y_test, y_pred3))
print("Accuracy for Naive Bayes:", metrics.accuracy_score(y_test, y_pred4))
