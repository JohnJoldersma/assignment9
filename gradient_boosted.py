import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import joblib

# train_data = pd.read_csv("snli_small_train.csv")
# test_data = pd.read_csv("snli_small_test.csv")

train_data = pd.read_csv("snli_train.csv")
test_data = pd.read_csv("snli_test.csv")

# Encode targets
le = LabelEncoder()
y_train = le.fit_transform(train_data['gold_label'])
X_train = train_data[['sentence1', 'sentence2']]

y_test = le.transform(test_data['gold_label'])
X_test = test_data[['sentence1', 'sentence2']]


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key): self.key = key
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.key]


pipeline = Pipeline([
    ('features', FeatureUnion([
        ('t1', Pipeline([('sel', TextSelector('sentence1')), ('tfidf', TfidfVectorizer())])),
        ('t2', Pipeline([('sel', TextSelector('sentence2')), ('tfidf', TfidfVectorizer())]))
    ])),
    ('clf', XGBClassifier())
])

param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 5, 7],
    'clf__min_child_weight': [1, 5, 10]
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc', # Choose an appropriate scoring metric
    cv=kfold,
    verbose=1,
    n_jobs=-1 # Use all available processors for parallel processing
)

# 4. Train/Test
pipeline.fit(X_train, y_train)
print(f"Accuracy: {pipeline.score(X_test, y_test)}")

joblib.dump(pipeline, 'xgb_pipeline.joblib')

model = pipeline.named_steps['clf']

loaded_pipeline = joblib.load('xgb_pipeline.joblib')

y_pred = loaded_pipeline.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# 5. Visualize the confusion matrix (recommended for better insight)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Class 0', 'Class 1'])
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Alternatively, use a classification report for other metrics like precision, recall, F1-score

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
