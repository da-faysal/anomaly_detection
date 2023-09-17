import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Embedding, Bidirectional, RepeatVector, TimeDistributed, Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Make the prediction reproducible
keras.utils.set_random_seed(43)

# Read the non-anomalous data, the model will be trained with
non_anom_df = pd.read_csv("traces_acronyms.txt", header=None).rename(columns={0: "data"})
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

# Create and train the BiLSTM Autoencoder
# Initialize the sequential model
model = keras.Sequential([

    # Input layer
    Input(shape=(max_sequence_length,)),

    # Embedding layer
    # input_dim: Vocabulary size, which is the number of unique tokens + 1
    # output_dim: Embedding dimension, 128 in this case
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64),

    # First BiLSTM layer with 32 neurons with 'relu' activation function
    Bidirectional(LSTM(32, activation="tanh", return_sequences=True)),
    Dropout(0.1),

    # Second BiLSTM layer with 32 neurons with 'relu' activation function
    Bidirectional(LSTM(16, activation="tanh", return_sequences=False)),
    Dropout(0.1),

    # RepeatVector layer repeats the input sequence the specified number of times (max_sequence_length)
    RepeatVector(max_sequence_length),

    # Third BiLSTM layer
    Bidirectional(LSTM(16, activation="tanh", return_sequences=True)),
    Dropout(0.1),

    # Fourth BiLSTM layer
    Bidirectional(LSTM(32, activation="tanh", return_sequences=True)),
    Dropout(0.1),

    # TimeDistributed layer with a Dense layer that applies a dense (fully connected) layer to each time step of the sequence
    TimeDistributed(Dense(1))
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Compile and fit the model for 40 epochs with batch size of 156
EPOCHS = 20
early_stopping = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
hist = model.fit(padded_text, padded_text, epochs=EPOCHS, batch_size=32, callbacks=[early_stopping], validation_split=0.2)
# Create df off loss and accuracy
hist_df = pd.DataFrame({
    "loss":hist.history["loss"],
    "val_loss":hist.history["val_loss"]
})
# hist_df.accuracy = hist_df.accuracy*100
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

# Input the trace to check
traces_to_check = pd.read_csv("Log_five.txt", header=None).rename(columns={0:"data"})
traces_to_check = traces_to_check.data.str.lower().str.strip()

# Calculate reconstruction error
recons_df = []
for tr in traces_to_check:
    recons_df.append(cal_reconstruction_error(tr))
recons_df = pd.concat(recons_df).reset_index(drop=True)

# Set a threshold for anomaly detection, one-tenth std above the mean
threshold = round(recons_df.reconstruction_error.mean() + 0.1*recons_df.reconstruction_error.std(), 2)

# Flagging anomalies
recons_df["anomaly"] = np.where(recons_df.reconstruction_error>threshold, 1, 0)

# Calculate the train accuract
def calculate_train_accuracy():
    # Use the trained model to predict the reconstructed (non-anomalous) data
    reconstructed_data = model.predict(padded_text)

    # Calculate the MSE between the original and reconstructed data
    mse = np.mean(np.power(padded_text - np.squeeze(reconstructed_data), 2), axis=1)

    # Calculate the accuracy based on a threshold (you can adjust the threshold as needed)
    threshold = np.mean(mse)+0.1*np.std(mse)
    is_normal = mse <= threshold
    accuracy = np.mean(is_normal)*100.0
    return accuracy

print(f"\nAccuracy after {hist_df.shape[0]} epochs: {calculate_train_accuracy()}")

# Print anomalies
print(f"\nDetected anomalies for threshold {threshold}:")
print(f"\nPrinting Anomalies:\n")
print(recons_df.query("anomaly==1").to_string())

# Save the anomaly score
recons_df.query("anomaly==1").to_csv("lstm_recons_error.csv", index=None)



# Plot the loss and accuracy
hist_df.set_index("epoch").plot(title="Train vs Val MSE", ylabel="MSE")
plt.savefig("lstm_losses.jpg")

# Plot the reconstruction error
recons_df.query("anomaly==1").drop("anomaly", axis=1).plot.bar(x="trace", y="reconstruction_error", title="Anomalous Reconstruction Error")
plt.savefig("lstm_ano_recons_error.jpg")

recons_df.query("anomaly==0").drop("anomaly", axis=1).plot.bar(x="trace", y="reconstruction_error", title="Non-anomalous Reconstruction Error")
plt.savefig("lstm_nonano_recons_error.jpg")