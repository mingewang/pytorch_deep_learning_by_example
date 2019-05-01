# sample code for text generate using RNN/lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

import numpy as np
import random
import sys
import io
import os

# a RNN class
class TextRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(TextRNN, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        # time steps: max_length for text
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # RNN input feature size: , hidden_size : 128
        self.basic_rnn = nn.LSTM(self.n_inputs, self.n_neurons)
        # we output feature size should be the same as input feature size
        # we need to use one-hot in RNN class, but not in data
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    #lstm need: h_n, c_n
    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons), torch.zeros(1, self.batch_size, self.n_neurons))

    def forward(self, X):
        # the original input is: batch_size x n_steps x n_inputs
        # as RNN need: (seq_len, batch_size, input_size):
        # so we use permute to
        # transforms X to new shape: n_steps X batch_size X n_inputs
        # permute (1,0,2) means
        # new data in dimension 0 comes from original dimension 1 etc.
        # thus (0,1,2) => (1, 0, 2)
        X = X.permute(1, 0, 2)

        # in case batch_size changed
        self.batch_size = X.size(1)

        self.hidden, self.cn = self.init_hidden()
        # by default those tensor are on cpu
        # let's sync hidden, cn to the same device as X
        device = X.device
        self.hidden, self.cn = self.hidden.to(device), self.cn.to(device)
 
        # now feed into rnn
        lstm_out,(self.hidden, self.cn) = self.basic_rnn(X, (self.hidden, self.cn))
        # we use hidden layer as input to the last linear layer
        out = self.FC(self.hidden)
        # finally output as output dimension's classes
        out =  out.view(-1, self.n_outputs) # batch_size X n_output
        return F.log_softmax(out, dim=1) # batch_size X n_output


# prepare data
# write a dataset
class TextDataset(Dataset):
  def __init__(self, root_dir, url, maxlen, download=False, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    filename = "tmp_text_dataset.txt"

    if download:
      download_url(url, root_dir, filename, md5=None)

    path = os.path.join(root_dir, filename)
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()

    #replace \n with space
    text=text.replace('\n',' ')
    print('corpus length:', len(text))

    # that will be our input/output features
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    # convert chars to integer category
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    # cut the text in semi-redundant sequences of maxlen characters
    self.maxlen = maxlen #40
    step = 1
    sentences = []
    next_chars = []
    # we prepare input and target output
    # we use input[i:i+maxlen1] to predict output[i + maxlen]
    for i in range(0, len(text) - maxlen, step):
        # i is the sample index
        # sentences contains the sample input
        # next_chars contains the target output
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('number of sequences:', len(sentences))

    print('Vectorization...')
    # one-hot vector for input: from first to last letter ( max_len)
    x = torch.zeros(len(sentences), maxlen, len(chars) )
    # should just be the predicted next char after max_len,
    y = torch.zeros(len(sentences), dtype=torch.long )
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        #y[i, char_indices[next_chars[i]]] = 1
        y[i] = char_indices[next_chars[i]]
    print('Vectorization done')

    # store the mapping for future sampling
    self.chars = chars
    self.char_indices = char_indices
    self.indices_char = indices_char
    # convert x, y to tensor from numpy
    self.x = x
    self.y = y

  # size of the whole dataset
  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    item_x = self.x[idx]
    item_y = self.y[idx]

    if self.transform:
        item_x = self.transform(item_x)
        item_y = self.transform(item_y)

    return item_x, item_y

  # help function
  def input_feature_size(self):
    return len(self.chars)

  # same as input feature
  def output_feature_size(self):
    return len(self.chars)

  def convert_index_to_char(self, char_index):
    char = self.indices_char[char_index]
    return char

  def convert_char_to_index(self, char):
    char_index = self.char_indices[char]
    return char_index

# # cut the text in semi-redundant sequences of maxlen characters
N_STEPS = 40
# This is internal hidden_size for our RNN
N_NEURONS = 128
N_EPHOCS = 30
# set our batch size
BATCH_SIZE = 64

max_len = N_STEPS

# download and load training dataset
url='http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/male.txt'
root_dir = "./data"
# transform                                                                     
#transform = transforms.Compose( [ transforms.ToTensor() ] )
trainset = TextDataset(root_dir, url, max_len )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# the input feature depends on the text we downloaded
N_INPUTS = trainset.input_feature_size()
# final output's feature dimension is the same as input dimension
N_OUTPUTS = trainset.output_feature_size()
# for simplicity reason, no test set

# just try to load one sample data
dataiter = iter(trainloader)
text_x, labels = dataiter.next()
#print("text_x is: ", text_x, " labels is: ", labels)
print("text_x is: ", text_x.shape, " labels is: ", labels.shape)

model = TextRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
#logits = model(text_x.view(-1, N_STEPS, N_INPUTS))
logits = model(text_x)

#print(logits[0:10])
print("output shape:", logits.shape)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# save our model here
model_saved_filename = "lstm_text_trained.model"

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

def train():
  # Model instance
  model = TextRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)

  # put model into that GPU or CPU
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training process
  for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
      train_running_loss = 0.0
      train_acc = 0.0
      # set model into training mode
      model.train()

      # TRAINING ROUND, get one batch of data
      for i, data in enumerate(trainloader):
           # zero the parameter gradients
          optimizer.zero_grad()

          # reset hidden states
          model.hidden = model.init_hidden()

          # get the inputs,
          # inputs will become (batch, N_STPES , N_INPUTS) e.g: 64 x 40 x len(chars)
          inputs, labels = data
          # data is python list, where inputs, labels are tensors
          # put data into that device: GPU or CPU
          inputs, labels = inputs.to(device), labels.to(device)
          # inputs will become (batch, h , w) e.g: 64 x 28 x 28
          # inputs = inputs.view(-1, N_STEPS, N_INPUTS)

          # forward + backward + optimize
          outputs = model(inputs)

          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # add loss for this run
          train_running_loss += loss.detach().item()
          train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

      # trained for this epoch
      print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
            %(epoch, train_running_loss / i, train_acc/i))

      # save model
      torch.save(model.state_dict(), model_saved_filename)


def generate_text(model, dataset, maxlen,seed_text):
    print('----- Generating text')
    generated = ''
    sentence = seed_text
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')

    input_feature_size = dataset.input_feature_size()

    for i in range(400):
        # sample sentence input
        x_pred = torch.zeros(1, maxlen, input_feature_size)
        # one-hot encode previous sentence
        for t, char in enumerate(sentence):
            char_index = dataset.convert_char_to_index(char)
            x_pred[0, t, char_index] = 1.
        # feed to model
        #import pdb
        #pdb.set_trace()
        # feed sentence into our model to predict next char
        output = model(x_pred)
        # here we just choose top 1, 
        #topv is the prob, topi is the index
        topv, topi = output.topk(1) 
        topi = topi[0][0]    
        next_index = topi.item() 
        # convert char_index into char
        next_char = dataset.convert_index_to_char(next_index)

        # append to our generated text
        generated += next_char
        # move/shift sentence to next one for the next round of prediction
        sentence = sentence[1:] + next_char
    print('----- Final text: "' + generated + '"')


train()

# load saved model 
model = TextRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
model.load_state_dict(torch.load(model_saved_filename))
model.eval()
generate_text(model, trainset, max_len, "her christorpher christos christy chrisy")

