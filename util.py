import os
import json
from torch.utils.data import DataLoader
from collections import Counter
import torch
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
import torch.nn as nn

dir_path = 'cornell movie-dialogs corpus'
convs_path = 'movie_conversations.txt'
lines_path = 'movie_lines.txt'

# makes dataset of lines and answer

def getData():

    """
    Returns a list of (text, label) pairs
    """

    convs = []

    with open(os.path.join(dir_path, convs_path), 'r', encoding= 'iso-8859-1') as file:

        for line in file:
            li = line.split('+++$+++')
            li = li[3]
            convs.append(eval(li)) 

    lines = dict()

    with open(os.path.join(dir_path, lines_path), 'r', encoding = 'iso-8859-1') as file:

        for line in file:
            li = line.split(' +++$+++ ')
            lines[li[0]] = li[4].strip('\n')

    data = []

    for conv in convs:
        for i in range(len(conv) - 1):
            li = (lines[conv[i]], lines[conv[i + 1]])
            data.append(li)

    return data

#builds a vocabulary object from torchtext

def tokenizeData(data):

    tokenizer = get_tokenizer('basic_english')

    tokenized_data = []

    for (text, label) in data:
        tokenized_data.append((tokenizer(text), tokenizer(label)))

    return tokenized_data

def buildVocab(data):

    """
    builds a Vocab object with pretrained embeddings, which can be used later 
    in the decoder
    """

    counter = Counter()
    tokenizer = get_tokenizer('basic_english')

    for sentence, response in data:
        counter.update(sentence)
        counter.update(response)


    vocab = Vocab(counter, 
            min_freq=3, 
            vectors='fasttext.simple.300d'
            )

    return vocab

def MapData(data, vocab):

    mappedData = []

    for sentence, response in data:
        mappedSentence = torch.tensor([vocab[i] for i in sentence] + [vocab['<eos>']])
        mappedResponse = torch.tensor([vocab[i] for i in response] + [response['<eos>']])

        mappedData.append((mappedSentence, mappedResponse))

    return mappedData

def UnmapData(mappedData, vocab):
    
    unmappedData = []

    for(sentence, response) in mappedData:
        unmappedSentence = [vocab.itos[i] for i in sentence] 
        unmappedResponse = [vocab.itos[i] for i in response]

        unmappedData.append((unmappedSentence, unmappedResponse))

    return unmappedData


class Encoder(nn.Module):

    def __init__(self, vocab):
        super(Encoder, self).__init__()

        self.vocabLen = len(vocab)
        
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.lstm1 = nn.GRU(300, 100, 1)  

    def forward(self, x):
        x = self.embedding(x)
        x, h = self.lstm1(x)
        return x, h
    

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, ):
        super(Decoder, self).__init__()

        self.lstm1 = nn.GRU(300, 100, 1)



