import torch
from util import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# TODO use pretrained word embeddings

data = getData()

tokenized_data = tokenizeData(data)

vocab = buildVocab(tokenized_data)

mappedData = MapData(tokenized_data, vocab)

encoder = Encoder(vocab)
test_data = mappedData[0][0]

x, h = encoder(torch.unsqueeze(test_data, 0))

print(h.shape)
