import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
# from tensorflow.keras.models import Model
from keras.layers import *
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error

from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import TextVectorization
from scipy.sparse import hstack

Model = tf.keras.models.Model

# train_data = pd.read_csv("snli_train.csv")
# test_data = pd.read_csv("snli_test.csv")
train_data = pd.read_csv("snli_small_train.csv")
test_data = pd.read_csv("snli_small_test.csv")

y_train = train_data['gold_label']
y_test = test_data['gold_label']

X_train_text1 = train_data['sentence1'].tolist()
X_train_text2 = train_data['sentence2'].tolist()
y_train_text = train_data['gold_label'].tolist()

X_train_dict1 = train_data['sentence1'].to_dict()
X_train_dict2 = train_data['sentence2'].to_dict()

vectorizer1 = TfidfVectorizer()
vectorizer2 = TfidfVectorizer()
y_vectorizer = TfidfVectorizer()

X_train1 = vectorizer1.fit_transform(train_data['sentence1'])
X_train2 = vectorizer2.fit_transform(train_data['sentence2'])
y_train_vector = y_vectorizer.fit_transform(train_data['gold_label'])
X_train = hstack([X_train1, X_train2])
X_test1 = vectorizer1.transform(test_data['sentence1'])
X_test2 = vectorizer2.transform(test_data['sentence2'])
X_test = hstack([X_test1, X_test2])

max_tokens = 10000  # Maximum vocabulary size
output_sequence_length = 50

vectorize_layer_1 = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)
vectorize_layer_1.adapt(train_data['sentence1'])

# TextVectorization layer for feature 2
vectorize_layer_2 = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_sequence_length
)
vectorize_layer_2.adapt(train_data['sentence2'])

input_1 = Input(shape=(1,), dtype=tf.string, name='sentence1', sparse=True)
input_2 = Input(shape=(1,), dtype=tf.string, name='sentence2', sparse=True)

# Process input 1
x1 = vectorize_layer_1(input_1)
x1 = Embedding(max_tokens, 64)(x1)
x1 = GlobalAveragePooling1D()(x1)

# Process input 2
x2 = vectorize_layer_2(input_2)
x2 = Embedding(max_tokens, 64)(x2)
x2 = GlobalAveragePooling1D()(x2)

# Concatenate the processed features
combined = Concatenate(axis=1)([x1, x2])

# Final layers
z = Dense(16, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(z) # Example for binary classification

# Create the model
model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# process_x1 = vectorize_layer_1(np.array(X_train_text1).astype(np.float32))
# process_x2 = vectorize_layer_1(np.array(X_train_text2).astype(np.float32))

# model.fit([process_x1, process_x2], y_train_text, epochs=10, batch_size=32)
# model.fit([x1, x2], y_train, epochs=10, batch_size=32)
# model.fit({'sentence1': X_train_dict1, 'sentence2': X_train_dict2}, y_train, epochs=10)

text_feature1_processed = vectorize_layer_1(test_data['sentence1'])
text_feature2_processed = vectorize_layer_2(test_data['sentence2'])

# text_feature1_np = np.array(text_feature1_processed)
# text_feature2_np = np.array(text_feature2_processed)

text_feature1_np = np.array(text_feature1_processed)
text_feature2_np = np.array(text_feature2_processed)

model.fit(
    x=[text_feature1_np, text_feature2_np],
    y=y_train_text,
    epochs=10,
    batch_size=32
)

model.save("NN_model.h5")

