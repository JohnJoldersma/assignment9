import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train_data = pd.read_csv("snli_train.csv")
test_data = pd.read_csv("snli_test.csv")

feature_columns = ['sentence1', 'sentence2']
X_train = train_data[feature_columns]
y_train = train_data['gold_label']
X_test = test_data[feature_columns]
y_test = test_data['gold_label']

# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(train_data[feature_columns])
# X_test = vectorizer.transform(test_data[feature_columns])

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf1', TfidfVectorizer(), 'sentence1'),
        ('tfidf2', TfidfVectorizer(), 'sentence2')
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classify', DecisionTreeClassifier())
])
pipeline.fit(X_train, y_train)

# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
#
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
