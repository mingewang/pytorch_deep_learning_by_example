# pytorch sample to show word embedding 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from numpy import array
from torchnlp.text_encoders import WhitespaceEncoder
from torchnlp.utils import pad_tensor
import matplotlib.pyplot as plt

# doc/words and its label
docs = ['China',
    'Italy',
    'Germany',
    'USA',
    'Canada',
    'Beijing',
    'Rome',
    'Berlin',
    'Washington DC',
    'Ottawa']

# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])


# we use integer to encode/represent the documents's word
# here we use torchnlp's Tokenizer
t  = WhitespaceEncoder( docs )
# t.vocab

# encode the whole document
encoded_docs =[ t.encode(x) for x in docs]
print("encoded_docs is:")
print(encoded_docs)
# encoded_docs will look this
#[tensor([5]),
# tensor([6]),
# tensor([7]),
## tensor([8]),
# tensor([9]),
# tensor([10]),
# tensor([11]),
# tensor([12]),
# tensor([13, 14]),
# tensor([15])

# Each doc/sequences could have different lengths
# and pytorch prefers 
# all inputs to have the same length.
# here we will pad all input sequences to have the length of 2
max_length = 2
padded_docs = [ pad_tensor(x, max_length) for x in encoded_docs ]
print("padded_docs is:")
print(padded_docs)

# now define our DNN module:
# we use nn.Embedding, then a linear layer with sigmoid
# for our binary classification
class MyNet(torch.nn.Module):
  # num_embeddings : vocab_size
  # embedding_dim:  the size of each embedding vector
  # max_length: the input max length
  # n_out: final out, here we set 1 
  # as we will use sigmoid for our binary classification
  def __init__(self, num_embeddings, embedding_dim, max_length, n_out):
    super(MyNet, self).__init__()
# nn.embedding layers
# first parameter: size of the vocabulary
# second parameter: output_dim, Dimension of the dense embedding.
# We will choose a small embedding space of 2 dimensions for easy plotting
    self.em = torch.nn.Embedding(num_embeddings, embedding_dim)
    # n_out is 1 here for binary classification
    self.linear_1 = torch.nn.Linear( max_length * embedding_dim, n_out)

  def forward(self, x):
    y = self.em(x)
    # need to flatten/squeeze, but keep the first dimension ( batch ) the same
    y = y.view(y.size()[0], -1)
    y = self.linear_1(y)
    y= torch.sigmoid(y)
    return y

  # helper function, embedding encode a token (x) for easy plot
  def embedding_encode(self, x):
    return self.em(x)

# let's create a model
num_embeddings = t.vocab_size
embedding_dim = 2
n_out = 1  # two classes
model = MyNet( num_embeddings, embedding_dim, max_length, n_out)

# Construct the loss function
# for binary classification
criterion = nn.BCELoss()
# Construct the optimizer,
# learning rate (lr) need trial-error
# if lr too small, learning will be very slow 
optimizer = optim.Adam(model.parameters(), lr=0.8 )

epochs = 5 
batch_size = 2 

# convert from list of tensor to pure tensor (*,2)
# so x_train is: [ [s11,s12], [s21,s22], ...]
x_train = torch.stack(padded_docs)
# make sure y_train the same as input
# so y_train is: [ [y1], [y2], ...]
y_train = torch.from_numpy(labels).float().view(-1,1)

model.train()
# training process, calculate Gradient Descent
for epoch in range(epochs):
    i = 0;
    print('epoch: ', epoch,' begin .. ')
    # batch feeding the data
    for i in range(0, x_train.size()[0], batch_size):

      # start debugger
      #import pdb; pdb.set_trace()

      # get a batch of data
      x = x_train[i:i+batch_size]
      y = y_train[i:i+batch_size]

      # Forward pass: Compute predicted y by passing x to the model
      y_pred = model(x)

      # Compute and print loss
      loss = criterion(y_pred, y)
      print('i: ', i,' loss: ', loss.item() )

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()

      # perform a backward pass (back propagation)
      loss.backward()

      # Update the parameters
      optimizer.step()

# Let's try to plot the what is the embedding
labels = []
data_x = []
data_y = []

model.eval()
# plot all the words we learned
for k in t.vocab:
  # encode using our token ( from string to int)
  encoded_k = t.encode(k)
  labels.append(k)
  # now let's use embedding to encode int to its representation
  tmp = model.embedding_encode(encoded_k)
  x = tmp[0][0].item()
  y = tmp[0][1].item()
  data_x.append( x )
  data_y.append( y )
  print("city is:", k, " encoded as:", encoded_k, " in embedding space, x=", x, ", y=",y)

plt.plot(data_x, data_y, 'ro')

# add label to each word point in the (x,y) space
for label, x, y in zip(labels, data_x, data_y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
