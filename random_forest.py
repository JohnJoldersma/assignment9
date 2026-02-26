import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = pd.read_csv("snli_small_train.csv")
test_data = pd.read_csv("snli_small_test.csv")

vec1 = CountVectorizer()
vec2 = CountVectorizer()
X1 = vec1.fit_transform(train_data['sentence1'])
X2 = vec2.fit_transform(train_data['sentence2'])

# X1_test = vec1.fit_transform(test_data['sentence1'])
# X2_test = vec2.fit_transform(test_data['sentence2'])

X = hstack([X1, X2])
# X_test = hstack([X1_test, X2_test])

y_test = test_data['gold_label']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, train_data['gold_label'])

with open('vectorizer1.pkl', 'wb') as f:
    pickle.dump(vec1, f)
with open('vectorizer2.pkl', 'wb') as f:
    pickle.dump(vec2, f)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('vectorizer1.pkl', 'rb') as f:
    loaded_vectorizer1 = pickle.load(f)
with open('vectorizer2.pkl', 'rb') as f:
    loaded_vectorizer2 = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

X_new_vectorized1 = loaded_vectorizer1.transform(test_data['sentence1'])
X_new_vectorized2 = loaded_vectorizer2.transform(test_data['sentence2'])
X_test = hstack([X_new_vectorized1, X_new_vectorized2])

y_pred = loaded_model.predict(X_test)

# 6. Evaluate the model and display accuracy results
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
