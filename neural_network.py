import pandas as pd
from keras.models import Sequential
from tensorflow.keras.models import Model
from keras.layers import *
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

train_data = pd.read_csv("snli_train.csv")
test_data = pd.read_csv("snli_test.csv")

y_train = train_data['gold_label']
y_test = test_data['gold_label']

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

input_1 = Input(shape=(1,), dtype=tf.string, name='sentence1')
input_2 = Input(shape=(1,), dtype=tf.string, name='sentence2')

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

model.fit([X_train1, X_train2], y_train_vector, epochs=10, batch_size=32)
# model.fit([x1, x2], y_train, epochs=10, batch_size=32)


model.save("NN_model.h5")

