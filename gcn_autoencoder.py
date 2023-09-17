# Import required modules
import pandas as pd
from tensorflow import keras
from keras.layers import Dropout
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Make the prediction reproducible
keras.utils.set_random_seed(43)

# Read the non-anomalous data, the model will be trained with
non_anom_df = pd.read_csv("traces_acronyms.txt", header=None).rename(columns={0:"data"})
non_anom_df.data = non_anom_df.data.str.lower().str.strip()

# Tokenize and encode the text data
# Create a Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer()

# Fit the Tokenizer on the non-anomalous text data
tokenizer.fit_on_texts(non_anom_df.data)

# Encode the non-anomalous text data using the learned vocabulary during the fit
encoded_text = tokenizer.texts_to_sequences(non_anom_df.data)

# Find the maximum sequence length in the encoded text data
max_sequence_length = max(len(seq) for seq in encoded_text)

# Pad the encoded text sequences to have the same length
padded_text = keras.preprocessing.sequence.pad_sequences(encoded_text, maxlen=max_sequence_length, padding="post")


# Create the gcn encoder model
input_layer = keras.layers.Input(shape=(max_sequence_length,))
embedding_layer = keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128)(input_layer)

# Create a fully connected graph (each word connected to every other word)
adjacency_matrix = keras.layers.Dot(axes=(2, 2))([embedding_layer, embedding_layer])

# Flatten the adjacency matrix to connect with the embeddings
adjacency_matrix = keras.layers.Flatten()(adjacency_matrix)
adjacency_matrix = keras.layers.Reshape((max_sequence_length, max_sequence_length))(adjacency_matrix)

# Graph Convolutional Layer
# Performs a dot product operation between the adjacency_matrix tensor and the embedding_layer tensor.
gcn_layer = keras.layers.Dot(axes=(2, 1))([adjacency_matrix, embedding_layer])

# Flattens the o/p of gcn layer
gcn_layer = keras.layers.Flatten()(gcn_layer)

# encoded representation of the input data.
encoded_output = keras.layers.Dense(64, activation="relu")(gcn_layer)

# input to the decoder part of the autoencoder.
decoder_input = keras.layers.Input(shape=(64,))

# output of the decoder part of the autoencoder.
decoder_output = keras.layers.Dense(128, activation="relu")(decoder_input)

# final output of the autoencoder.
decoded_output = keras.layers.Dense(max_sequence_length, activation="relu")(decoder_output)

# Create the Full gcn Autoencoder Model
# encoder part of the gcn autoencoder
encoder_model = keras.Model(inputs=input_layer, outputs=encoded_output)

# decoder part of the gcn autoencoder
decoder_model = keras.Model(inputs=decoder_input, outputs=decoded_output)

# full GCN autoencoder model
autoencoder_input = keras.layers.Input(shape=(max_sequence_length,))

# applies the encoder model to the input data 
encoded_data = encoder_model(autoencoder_input)

# applies the decoder modell to the encoded representation to reconstruct the input data
decoded_data = decoder_model(encoded_data)

# full gcn autoencoder which takes input data and produces the reconstructed data. It combines both the encoder and decoder models
model = keras.Model(inputs=autoencoder_input, outputs=decoded_data)

# Calculate reconstruction error
def cal_reconstruction_error(text_data):
    # Encode the anomalous data using the tokenizer
    encoded_data = tokenizer.texts_to_sequences([text_data])

    # Pad the encoded data to match the model's input shape
    padded_data = keras.preprocessing.sequence.pad_sequences(encoded_data, maxlen=max_sequence_length, padding='post')

    # Use the trained model to predict the reconstructed (anomalous) data
    reconstructed_data = model.predict(padded_data)

    # Calculate the Mean Squared Error (MSE) between the original and reconstructed data
    mse = np.mean(np.power(padded_data - np.squeeze(reconstructed_data), 2), axis=1)
    
    # Create a df off reconstruction error and corresponsing text
    recons_df = pd.DataFrame({
        "trace":text_data,
        "reconstruction_error":mse
    }, index=[0])
    
    return recons_df


# Compile and Train the model for 30 epochs
EPOCHS = 50
early_stopping = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
hist = model.fit(padded_text, padded_text, epochs=EPOCHS, batch_size=64, callbacks=[early_stopping], validation_split=0.2)

# Create df off loss and accuracy
hist_df = pd.DataFrame({
    "train_loss":hist.history["loss"],
     "val_loss":hist.history["val_loss"]
})
hist_df["epoch"] = range(1, hist_df.shape[0]+1)


# Calculate reconstruction error
def cal_reconstruction_error(text_data):
    # Encode the anomalous data using the tokenizer
    encoded_data = tokenizer.texts_to_sequences([text_data])

    # Pad the encoded data to match the model's input shape
    padded_data = keras.preprocessing.sequence.pad_sequences(encoded_data, maxlen=max_sequence_length, padding='post')

    # Use the trained model to predict the reconstructed (anomalous) data
    reconstructed_data = model.predict(padded_data)

    # Calculate the Mean Squared Error (MSE) between the original and reconstructed data
    mse = np.mean(np.power(padded_data - np.squeeze(reconstructed_data), 2), axis=1)
    
    # Create a df off reconstruction error and corresponsing text
    recons_df = pd.DataFrame({
        "trace":text_data,
        "reconstruction_error":mse
    }, index=[0])
    
    return recons_df

# Input the trace to check, always be in list
# Input the trace to check
traces_to_check = pd.read_csv("Log_five.txt", header=None).rename(columns={0:"data"})
traces_to_check = traces_to_check.data.str.lower().str.strip()

# Calculate reconstruction error
recons_df = []
for tr in traces_to_check:
    recons_df.append(cal_reconstruction_error(tr))
recons_df = pd.concat(recons_df).reset_index(drop=True)

# Set a threshold for anomaly detection,
threshold = round(recons_df.reconstruction_error.mean()+0.001*recons_df.reconstruction_error.std(), 2)

# Flagging anomalies
recons_df["anomaly"] = np.where(recons_df.reconstruction_error>threshold, 1, 0)

# Print anomalies
print(f"\nDetected anomalies for threshold {threshold}:")
print(recons_df.query("anomaly==1").to_string())

# Save the score
recons_df.query("anomaly==1").to_csv("gcn_recons_error.csv", index=None)


# Plot the loss and accuracy
hist_df.set_index("epoch").plot(title="Train vs Val MSE", ylabel="MSE Loss")
plt.savefig("gcn_losss.jpg")


# Plot the reconstruction error
recons_df.query("anomaly==1").drop("anomaly", axis=1).plot.bar(x="trace", y="reconstruction_error", title="Anomalous Reconstruction Error")
plt.savefig("gcn_ano_recons_error.jpg")

recons_df.query("anomaly==0").drop("anomaly", axis=1).plot.bar(x="trace", y="reconstruction_error", title="Non-anomalous Reconstruction Error")
plt.savefig("gcn_nonano_recons_error.jpg")


# Plot the reconstruction error and compare them
lstm_error = pd.read_csv("lstm_recons_error.csv").drop("anomaly", axis=1).rename(columns={"reconstruction_error":"lstm_recons_error"})
gcn_error = pd.read_csv("gcn_recons_error.csv").drop("anomaly", axis=1).rename(columns={"reconstruction_error":"gcn_recons_error"})

# Plot line chart and save
fig, ax = plt.subplots(figsize=(10, 5))
fig = pd.merge(lstm_error, gcn_error, on="trace").set_index("trace").plot(ax=ax, ylabel="MSE", title="LSTM vs GCN Reconstruction Error")
plt.xticks(rotation=90)
plt.savefig("recons_line_chart.jpg", bbox_inches='tight')

# Plot bar chart and save
fig, ax = plt.subplots(figsize=(10, 5))
fig = pd.merge(lstm_error, gcn_error, on="trace").set_index("trace").plot.bar(ax=ax, ylabel="MSE", title="LSTM vs GCN Reconstruction Error")
plt.xticks(rotation=90)
plt.savefig("recons_bar_chart.jpg", bbox_inches='tight')