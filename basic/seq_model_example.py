# example to show pytorch's Sequential model
import torch
import torch.nn as nn

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 100, 64, 10, 32

# Create a model
# keras's dense layer =  nn.Linear + activation funcation  in pytorch
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h,n_out))

# Construct the loss function
criterion = torch.nn.MSELoss(reduction='sum') 

# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# prepare some data: 
# each data point has n_in-dimension input, and n_out-dimension output
num_training = 100
x_train =  torch.FloatTensor(num_training, n_in).uniform_()
y_train =  torch.FloatTensor(num_training, n_out).uniform_()

num_val = 10
x_val =  torch.FloatTensor(num_val, n_in).uniform_()
y_val =  torch.FloatTensor(num_val, n_out).uniform_()

epochs = 5

# when to check the validation
N_eval_epoch = 3;

# training process, calculate Gradient Descent
for epoch in range(epochs):
    i = 0;
    print('epoch: ', epoch,' begin .. ')
    # batch feeding the data
    for i in range(0, x_train.size()[0], batch_size):
      # get a batch of data
      x = x_train[i:i+batch_size]
      y = y_train[i:i+batch_size]

      # Forward pass: Compute predicted y by passing x to the model
      y_pred = model(x)

      # Compute and print loss
      loss = criterion(y_pred, y)
      print('i: ', i,' loss: ', loss.item())

      # Zero gradients, perform a backward pass, and update the weights.
      optimizer.zero_grad()

      # perform a backward pass (backpropagation)
      loss.backward()

      # Update the parameters
      optimizer.step()

    # Validation 
    if epoch%N_eval_epoch==0:
      with torch.no_grad(): 
        for i in range(0, x_val.size()[0], batch_size):
          x = x_val[i:i+batch_size]                                               
          y = y_val[i:i+batch_size]  
          y_pred = model(x)
          if torch.equal( y, y_pred ):
            print("validation: predict correct")
          else:
            print("validation: predict wrong")
      
