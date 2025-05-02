import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

data_file = '/Users/katyayni/Desktop/malwareanalysis/extractedopcodes.txt'

opcode_sequences = []

with open(data_file, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("Extracted opcodes:"):
            opcodes = line.strip().split('[')[-1].split(']')[0].replace("'", "").replace(" ", "").split(',')
            if opcodes:
                opcode_sequences.append(opcodes)


if len(opcode_sequences) < 268:
    raise ValueError("There should be at least 268 sequences in the file.")


goodware_sequences = opcode_sequences[:268]
malware_sequences = opcode_sequences[268:]

print(f"Total goodware sequences: {len(goodware_sequences)}")
print(f"Total malware sequences: {len(malware_sequences)}")

goodware_labels = [0] * len(goodware_sequences)
malware_labels = [1] * len(malware_sequences)

all_sequences = goodware_sequences + malware_sequences
all_labels = goodware_labels + malware_labels


w2v_model = Word2Vec.load("opcode_word2vec.model")


def embed_sequence(sequence, model, embedding_size):
    return [model.wv[opcode] if opcode in model.wv else np.zeros(embedding_size) for opcode in sequence]

embedding_size = w2v_model.vector_size
embedded_sequences = [embed_sequence(seq, w2v_model, embedding_size) for seq in all_sequences]

max_length = max(len(seq) for seq in embedded_sequences)
padded_sequences = pad_sequences(embedded_sequences, maxlen=max_length, dtype='float32', padding='post')

all_labels = to_categorical(all_labels, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, all_labels, test_size=0.2, random_state=42)


model = Sequential()
model.add(LSTM(64, input_shape=(max_length, embedding_size), return_sequences=False))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")



