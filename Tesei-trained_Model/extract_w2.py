"""
Title:   Extract Omega_2 from Tesei-trained PML model
Author:  Lilianna Houston, Ghosh Lab
Date:    July 22nd 2024
Purpose: This code extracts the omega_2 (w2) value of protein sequences from a ML 
         model trained on the Tesei 2023 dataset.
Inputs:  CSV of protein sequences and weights of the ML model.
Outputs: CSV of protein sequences with omega_2 predictions.
"""

# Enter path to desired weights
path_to_weights = "weights/weights_2.best.hdf5"
# Enter path to data file containing sequnces
path_to_data = "test.csv"
# Specify sequence column
seq_column = 3

# Import packages
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import sys

# Define a dictionary of amino acid residues (alphebetical by full name, 
# stored in letter representation) and their charge.
amino_acid_data = {
    "A": 0,  # Alanine
    "R": 1,  # Arginine
    "N": 0,  # Asparagine
    "D": -1, # Aspartic acid
    "C": 0,  # Cysteine
    "E": -1, # Glutamic acid
    "Q": 0,  # Glutamine
    "G": 0,  # Glycine
    "H": 0,  # Histidine
    "I": 0,  # Isoleucine
    "L": 0,  # Leucine
    "K": 1,  # Lysine
    "M": 0,  # Methionine
    "F": 0,  # Phenylalanine
    "P": 0,  # Proline
    "S": 0,  # Serine
    "T": 0,  # Threonine
    "W": 0,  # Tryptophan
    "Y": 0,  # Tyrosine
    "V": 0   # Valine
}

# Function to one-hot encode a protein sequence
def hotcode_seq(seq):
    hotcode_matrix = np.zeros((21, 1496))  
    for i in range(len(seq)):
        index = list(amino_acid_data.keys()).index(seq[i])  # Find the index of the amino acid
        hotcode_matrix[index, i] = 1  # Set the corresponding position in the matrix to 1
    hotcode_matrix[20, (i+1):] = 1  # Set remaining positions in the last row to 1
    return hotcode_matrix

# Function to convert a list of sequences to their one-hot encoded matrices
def make_hotcodes(data):
    hotcodes = []
    for i in range(len(data)):
        hotcodes.append(hotcode_seq(data[i]))
    return np.asarray(hotcodes)  

# -------------- Create a model framework in which to load weights ------------------- 

# The Tesei trained model uses a maximum sequences length of 1496, the length
# of the longest sequence in the Tesei 2023 dataset.
model_input_shape = (21, 1496, 1)  

image_input = keras.Input(shape=model_input_shape) 

# Convolutional layer with 29 filters, kernel size (21, 6), and ReLU activation
conv1 = layers.Conv2D(29, kernel_size=(21, 6), activation='relu')(image_input)

# Flatten the output from the convolutional layer
flatten = layers.Flatten()(conv1)

# Dense layer with 100 units and softsign activation
dense1 = layers.Dense(100, activation='softsign')(flatten)
# Dense layer with 30 units and softsign activation
dense2 = layers.Dense(30, activation='softsign')(dense1)
# Output layer with 1 unit and linear activation
output = layers.Dense(1, activation='linear')(dense2)

model = keras.Model(inputs=image_input, outputs=output, name="model")
# --------------------------------------------------------------------------------------

# Load pre-trained weights into model
model.load_weights(path_to_weights, skip_mismatch=False)

# Compile the model with Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Load data file
data = pd.read_csv(path_to_data)

# Extract the sequences from the data
seqs = data.iloc[:, seq_column]

# Convert the sequences to their one-hot encoded matrices
hots = make_hotcodes(seqs)

# Predict the output for the one-hot encoded sequences using the model
preds = model.predict(hots)

# Extract the predictions from the model output
w2_preds = preds[:, 0]

# Add the predictions to the data
data["w2_preds_tesei_model"] = w2_preds

# Save the data with predictions to a new CSV file
data.to_csv("test_w2preds.csv", index=False)
