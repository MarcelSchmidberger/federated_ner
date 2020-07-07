import torch
import numpy as np

def read_files(filenames, *, encoding="UTF8"):
    sentences = []
    for filename in filenames:
        with open(filename, mode='rt', encoding=encoding) as f:
            sentence = []
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                splits = line.split(' ')
                sentence.append([splits[0], splits[-1]])

            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

    return sentences

def format_to_tensor(sentences, word2Idx, label2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        labelIndices = []

        for word, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, labelIndices])

    return dataset