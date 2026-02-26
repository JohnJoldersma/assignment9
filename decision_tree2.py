import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_data = pd.read_csv("snli_train.csv")
test_data = pd.read_csv("snli_test.csv")

y_train = train_data['gold_label']
y_test = test_data['gold_label']

vectorizer1 = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()

X_train1 = vectorizer1.fit_transform(train_data['sentence1'])
X_train2 = vectorizer2.fit_transform(train_data['sentence2'])
X_train = hstack([X_train1, X_train2])
X_test1 = vectorizer1.transform(test_data['sentence1'])
X_test2 = vectorizer2.transform(test_data['sentence2'])
X_test = hstack([X_test1, X_test2])


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 4. Visualize the confusion matrix with labels
# You can use ConfusionMatrixDisplay for a simple plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.show()

# Alternatively, use a Seaborn heatmap for more customization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix using Seaborn')
plt.show()
