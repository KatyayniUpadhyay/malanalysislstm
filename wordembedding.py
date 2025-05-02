from gensim.models import Word2Vec
import os
import random
from malwarebenign import extract_opcode_sequence


def train_word2vec_model(opcode_sequences, vector_size=100, window=30, min_count=1, save_path=None):
    model = Word2Vec(sentences=opcode_sequences, vector_size=vector_size, window=window, min_count=min_count, sg=0)  # sg=0 for CBOW
    
    if save_path:
        model.save(save_path)
        print(f"Word2Vec model saved to {save_path}")
    
    return model


def split_files(files, train_ratio=0.7):
    random.shuffle(files)
    split_point = int(len(files) * train_ratio)
    return files[:split_point], files[split_point:]


def get_opcode_sequences(directory, train_ratio=0.7):
    files = os.listdir(directory)
    train_files, test_files = split_files(files, train_ratio)

    train_sequences = [extract_opcode_sequence(os.path.join(directory, file)) for file in train_files]
    test_sequences = [extract_opcode_sequence(os.path.join(directory, file)) for file in test_files]

    return train_sequences, test_sequences

    from gensim.models import Word2Vec
import numpy as np
import os
import random
from malwarebenign import extract_opcode_sequence

def extract_features(file_path, model_path='/Users/katyayni/Desktop/malwareanalysis/word2vec_model.model'):
    """Extracts Word2Vec features for a given file containing opcode sequences."""
    
    # Load the saved Word2Vec model
    model = Word2Vec.load(model_path)

    # Extract opcode sequence from the file
    opcode_sequence = extract_opcode_sequence(file_path)
    
    # Convert opcodes to Word2Vec embeddings
    feature_vector = []
    for opcode in opcode_sequence:
        if opcode in model.wv:
            feature_vector.append(model.wv[opcode])
    
    # If no features were extracted, return a zero vector
    if len(feature_vector) == 0:
        return np.zeros(model.vector_size)
    
    # Average the opcode vectors
    return np.mean(feature_vector, axis=0)



goodware_dir = '/Users/katyayni/Desktop/malwareanalysis/goodware'
malware_dir = '/Users/katyayni/Desktop/malwareanalysis/malware'


train_opcode_sequences_goodware, test_opcode_sequences_goodware = get_opcode_sequences(goodware_dir, train_ratio=0.7)
train_opcode_sequences_malware, test_opcode_sequences_malware = get_opcode_sequences(malware_dir, train_ratio=0.7)

train_opcode_sequences = train_opcode_sequences_goodware + train_opcode_sequences_malware
test_opcode_sequences = test_opcode_sequences_goodware + test_opcode_sequences_malware


y_train = [0] * len(train_opcode_sequences_goodware) + [1] * len(train_opcode_sequences_malware)
y_test = [0] * len(test_opcode_sequences_goodware) + [1] * len(test_opcode_sequences_malware)

save_path = '/Users/katyayni/Desktop/malwareanalysis/word2vec_model.model'
word2vec_model = train_word2vec_model(train_opcode_sequences, save_path=save_path)



