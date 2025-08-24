import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        at = tf.nn.softmax(et, axis=1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

import nltk
from nltk.corpus import names
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Layer

# Load the names dataset
male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]

# Combine and shuffle the dataset
all_names = male_names + female_names
random.shuffle(all_names)

# Separate the names and labels
names_data = [name for name, gender in all_names]
labels = [gender for name, gender in all_names]

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Tokenize the names
tokenizer = Tokenizer(char_level=True)  # Tokenize at the character level
tokenizer.fit_on_texts(names_data)
sequences = tokenizer.texts_to_sequences(names_data)
max_length = max(len(name) for name in names_data)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Build the model
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
embedding_dim = 64

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Attention())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)

# Function to predict the gender of a name
def predict_gender(name):
    name_sequence = tokenizer.texts_to_sequences([name])
    name_padded = pad_sequences(name_sequence, maxlen=max_length, padding='post')
    prediction = model.predict(name_padded)
    predicted_label = (prediction > 0.5).astype("int32")
    gender = label_encoder.inverse_transform(predicted_label)[0]
    return gender

# Test the function with some examples
test_names = ["Alice", "Bob", "Charlie", "Diana", "Ahmad", "Maytham", "Enas", "Rama", "Ronaldo","Kareem"]
for name in test_names:
    gender = predict_gender(name)
    print(f"The name '{name}' is predicted to be '{gender}'.")
