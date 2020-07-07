
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import read_files, format_to_tensor
from model import BiLSTM
import os
import numpy as np

EMBEDDING_DIM = 300

train_data_path = "/home/marcel/voize/voize-react-native/nlp/train"

train_sentences = read_files([os.path.join(train_data_path, p) for p in os.listdir(train_data_path)])

labels = set()
words = {}

print("Extracting words and labels...")
for sentence in train_sentences:
  for token, label in sentence:
    labels.add(label)
    words[token.lower()] = True
print(f"Extracted {len(words)} words and {len(labels)} labels.")

# mapping for labels
label2Idx = {}
for label in labels:
    label2Idx[label] = len(label2Idx)

# read GLoVE word embeddings
word2Idx = {}
word_embeddings = []

word2Idx["PADDING_TOKEN"] = 0
vector = np.zeros(EMBEDDING_DIM)
word_embeddings.append(vector)

word2Idx["UNKNOWN_TOKEN"] = 1
vector = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
word_embeddings.append(vector)

print("Loading embeddings...") 
embeddings = gensim.models.KeyedVectors.load_word2vec_format('../embeddings/german.model', binary=True)
print("Done.")

# loop through each word in embeddings
for word in embeddings.vocab:
    if word.lower() in words:
        vector = embeddings.wv[word]
        word_embeddings.append(vector)
        word2Idx[word] = len(word2Idx)

word_embeddings = np.array(word_embeddings)
print(f"Found embeddings for {word_embeddings.shape[0]} of {len(words)} words.")

train_sentences = format_to_tensor(train_sentences, word2Idx, label2Idx)

model = BiLSTM(word_embeddings=torch.FloatTensor(word_embeddings), num_classes=len(labels))
model.train()

epochs = 50
learning_rate = 0.015
momentum = 0.9

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def eval():
    correct = 0
    total = 0

    for tokens, true_labels in train_sentences:
        total += len(true_labels)
        tokens = torch.LongTensor([tokens])
        true_labels = torch.LongTensor(true_labels)
        predicted_labels = model(tokens)
        predicted_labels = predicted_labels.argmax(axis=2).squeeze(dim=0)
        correct += (predicted_labels == true_labels).sum().item()

    print("Overall accuracy: " + str(correct / total))

for epoch in range(epochs):
    for tokens, true_labels in train_sentences:
        tokens = torch.LongTensor([tokens])
        true_labels = torch.LongTensor(true_labels)
        optimizer.zero_grad()
        predicted_labels = model(tokens)
        predicted_labels = predicted_labels.squeeze(dim=0)
        loss = criterion(predicted_labels, true_labels)
        loss.backward()
        optimizer.step()

    print('Epoch %d | Loss: %.3f' % (epoch + 1, loss.item()))
    eval()


