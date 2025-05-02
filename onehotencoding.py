from collections import Counter
from malwarebenign import process_directory, extract_opcode_sequence 

def build_opcode_vocabulary(file_opcodes):
    # Flatten all opcode sequences from all files into a single list
    all_opcodes = [opcode for opcodes in file_opcodes.values() for opcode in opcodes]
    opcode_freq = Counter(all_opcodes)
    vocabulary = {opcode: idx for idx, (opcode, _) in enumerate(opcode_freq.items())}
    return vocabulary

def one_hot_encode(file_opcodes, vocabulary):
    vocab_size = len(vocabulary)
    one_hot_encoded_files = {}
    
    for filename, opcode_sequence in file_opcodes.items():
        one_hot_sequence = []
        for opcode in opcode_sequence:
            one_hot_vector = [0] * vocab_size
            if opcode in vocabulary:
                one_hot_vector[vocabulary[opcode]] = 1
            one_hot_sequence.append(one_hot_vector)
        one_hot_encoded_files[filename] = one_hot_sequence
    
    return one_hot_encoded_files


goodware_dir = '/Users/katyayni/Desktop/malwareanalysis/goodware'
malware_dir = '/Users/katyayni/Desktop/malwareanalysis/malware'

goodware_opcodes = process_directory(goodware_dir)
malware_opcodes = process_directory(malware_dir)

# Combine both goodware and malware opcodes
all_file_opcodes = {**goodware_opcodes, **malware_opcodes}

# Build vocabulary
vocabulary = build_opcode_vocabulary(all_file_opcodes)

# One-hot encode the sequences
one_hot_goodware = one_hot_encode(goodware_opcodes, vocabulary)
one_hot_malware = one_hot_encode(malware_opcodes, vocabulary)

# Print or process one-hot encoded data as needed
print(one_hot_goodware)
print(one_hot_malware)

