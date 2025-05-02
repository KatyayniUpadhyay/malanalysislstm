import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from wordembedding import train_opcode_sequences, test_opcode_sequences, y_test, y_train

# Custom LSTM cell implementation
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for input, forget, cell, and output gates
        self.W_f = np.random.randn(hidden_size, input_size)
        self.U_f = np.random.randn(hidden_size, hidden_size)
        self.b_f = np.zeros((hidden_size, 1))

        self.W_i = np.random.randn(hidden_size, input_size)
        self.U_i = np.random.randn(hidden_size, hidden_size)
        self.b_i = np.zeros((hidden_size, 1))

        self.W_c = np.random.randn(hidden_size, input_size)
        self.U_c = np.random.randn(hidden_size, hidden_size)
        self.b_c = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size)
        self.U_o = np.random.randn(hidden_size, hidden_size)
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x_t, h_prev, C_prev):
        # Forget gate
        f_t = sigmoid(np.dot(self.W_f, x_t) + np.dot(self.U_f, h_prev) + self.b_f)
        
        # Input gate
        i_t = sigmoid(np.dot(self.W_i, x_t) + np.dot(self.U_i, h_prev) + self.b_i)
        
        # Candidate cell state
        C_tilde_t = tanh(np.dot(self.W_c, x_t) + np.dot(self.U_c, h_prev) + self.b_c)
        
        # Cell state
        C_t = f_t * C_prev + i_t * C_tilde_t
        
        # Output gate
        o_t = sigmoid(np.dot(self.W_o, x_t) + np.dot(self.U_o, h_prev) + self.b_o)
        
        # Hidden state
        h_t = o_t * tanh(C_t)
        
        return h_t, C_t

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Loading the trained Word2Vec model
word2vec_model_path = '/Users/katyayni/Desktop/malwareanalysis/word2vec_model.model'
word2vec_model = Word2Vec.load(word2vec_model_path)

def get_opcode_vectors(opcode_sequence, model):
    vectors = []
    for opcode in opcode_sequence:
        if opcode in model.wv:
            vectors.append(model.wv[opcode])
    return vectors

# Preparing the data
X_train = [get_opcode_vectors(seq, word2vec_model) for seq in train_opcode_sequences]
X_test = [get_opcode_vectors(seq, word2vec_model) for seq in test_opcode_sequences]

def convert_and_pad_sequences(sequences, max_seq_length, embedding_dim):
    padded_sequences = []
    for seq in sequences:
        seq_array = np.array(seq)
        if seq_array.shape[0] < max_seq_length:
            padding = np.zeros((max_seq_length - seq_array.shape[0], embedding_dim))
            seq_array = np.vstack([seq_array, padding])
        elif seq_array.shape[0] > max_seq_length:
            seq_array = seq_array[:max_seq_length]
        padded_sequences.append(seq_array)
    return np.array(padded_sequences)

max_seq_length = 100
embedding_dim = 100  # Adjust this to match the dimensionality of your embeddings

X_train_padded = convert_and_pad_sequences(X_train, max_seq_length, embedding_dim)
X_test_padded = convert_and_pad_sequences(X_test, max_seq_length, embedding_dim)

# Initialize custom LSTM cells
hidden_size = 128
lstm_cell_1 = LSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
lstm_cell_2 = LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

# Manually process sequences through the custom LSTM
def process_sequences_with_lstm(sequences, lstm_cell):
    h = np.zeros((hidden_size, 1))
    C = np.zeros((hidden_size, 1))
    outputs = []
    for sequence in sequences:
        sequence_output = []
        for t in range(sequence.shape[0]):
            x_t = sequence[t].reshape(-1, 1)  # Reshape to match input size
            h, C = lstm_cell.forward(x_t, h, C)
            sequence_output.append(h)
        outputs.append(np.array(sequence_output).squeeze(axis=-1))
    return np.array(outputs)

# Processing training and test sequences through the first LSTM layer
X_train_processed = process_sequences_with_lstm(X_train_padded, lstm_cell_1)
X_test_processed = process_sequences_with_lstm(X_test_padded, lstm_cell_1)

# Processing sequences through the second LSTM layer
X_train_processed_2 = process_sequences_with_lstm(X_train_processed, lstm_cell_2)
X_test_processed_2 = process_sequences_with_lstm(X_test_processed, lstm_cell_2)

# Global Average Pooling (mean pooling) after LSTM layers
X_train_pooled = np.mean(X_train_processed_2, axis=1)
X_test_pooled = np.mean(X_test_processed_2, axis=1)

X_train_pooled = np.array(X_train_pooled)
X_test_pooled = np.array(X_test_pooled)

print(f"X_train_pooled shape: {X_train_pooled.shape}")
print(f"X_test_pooled shape: {X_test_pooled.shape}")

y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

input_shape = (X_train_pooled.shape[1],)

# Defining the final model using Keras
inputs = Input(shape=input_shape)
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)


model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train_pooled, y_train, epochs=10, batch_size=32)

# Evaluating the model
loss, accuracy = model.evaluate(X_test_pooled, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Get predictions for the custom LSTM
y_pred_custom = model.predict(X_test_pooled)  # Use the pooled test data from the custom LSTM
y_pred_custom = (y_pred_custom > 0.5).astype(int)  # Convert probabilities to binary values (0 or 1)

# Calculate F1-score
f1_custom = f1_score(y_test, y_pred_custom)
print(f"F1-Score for Custom LSTM: {f1_custom:.4f}")

# Save the trained model as an HDF5 file
model.save("custom_lstm_model.h5")
print("Model saved as custom_lstm_model.h5")
