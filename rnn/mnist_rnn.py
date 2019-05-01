# sample code for RNN to classify MNIST data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# MNIST data is 28x28
# instead of using CNN, we can actually try RNN as well:
# the first 28 can be thought as time steps
# the second 28 can be thought as input features
N_STEPS = 28
N_INPUTS = 28
# This is internal hidden_size for our RNN
N_NEURONS = 150
# final output since MNIST has 10 classes
N_OUTPUTS = 10
N_EPHOCS = 10

# a RNN class
class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        # time steps: 28
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # RNN input size: 28, hidden_size : 150 
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons) 
        # we output 10 classes 
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        
    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons))
        
    def forward(self, X):
        # the original input is: batch_size x n_steps x n_inputs
        # as RNN need: (seq_len, batch_size, input_size):
        # so we use permute to
        # transforms X to new shape: n_steps X batch_size X n_inputs
        # permute (1,0,2) means 
        # new data in dimension 0 comes from original dimension 1 etc.
        # thus (0,1,2) => (1, 0, 2)
        X = X.permute(1, 0, 2) 
        
        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()
        # now feed into rnn 
        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)      
        # we use hidden layer as input to the last linear layer
        out = self.FC(self.hidden)
        # finally output as  10 classes
        return out.view(-1, self.n_outputs) # batch_size X n_output

# set our batch size
BATCH_SIZE = 64
# transform
transform = transforms.Compose( [ transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)) ] )
# save download location, adjust to your env
data_dir="../data"

# download and load training dataset
trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

# download and load testing dataset
testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)


# just try to load one sample data
dataiter = iter(trainloader)
images, labels = dataiter.next()
model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
logits = model(images.view(-1, 28,28))
print(logits[0:10])

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instance
model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

# Training process
for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    # set model into training mode
    model.train()
    
    # TRAINING ROUND
    for i, data in enumerate(trainloader):
         # zero the parameter gradients
        optimizer.zero_grad()
        
        # reset hidden states
        model.hidden = model.init_hidden() 
        
        # get the inputs
        inputs, labels = data
        # inputs will become (batch, h , w) e.g: 64 x 28 x 28
        inputs = inputs.view(-1, 28,28) 

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
         
    
# now let's evaluate on test data
test_acc = 0.0
# set model into evaluation mode
model.eval()

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.view(-1, 28, 28)
    outputs = model(inputs)
    test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
        
print('Test Accuracy: %.2f'%( test_acc/i))
