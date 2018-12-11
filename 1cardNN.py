import torch
import time
import os

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights or other state.
"""
def toSoftmax(y):
  """Takes in an n x 1 x 4 x 13 vector of y and returns a n x 1 x 4.
  The 4x13 are assumed to all have the same value, and that value is 
  encoded as a one-hot in the 3rd dimension of the returned tensor"""
  onehot = torch.zeros(y.size()[0])
  for i in range(y.size()[0]):
    onehot[i] = (int)(y[i][0][0][0].item())%2
  return onehot

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# T is size of training set
# N is batch size; D_in is input dimension(the number of 4x13 layers);
# H is hidden dimension; poolSize is the number of outputs from maxPooling; D_out is output dimension.
T, N, D_in, H, poolSize, D_out = 100000, 200, 4, 13, 26*4*13, 1

# Create random Tensors to hold inputs and outputs
x_list = []
for i in range(0,100000):
  x_list.append(torch.load(os.path.join('trn', str(i)+'.pt')))
print("loaded training data")
x = torch.stack(x_list)
y = torch.index_select(x, 1, torch.tensor([4]))
y = toSoftmax(y)
x = torch.index_select(x, 1, torch.tensor(range(4)))
#x = torch.rand(N, D_in, device=device)s
#y = torch.rand(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(D_in, H, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(H, 2*H, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(poolSize, D_out)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, poolSize)
        x = self.fc1(x)
        return torch.sigmoid(x)

model = Net()
"""model = torch.nn.Sequential(
          torch.nn.Conv2d(D_in, H, 3, stride=1, padding = 1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
          torch.nn.Linear(26*N, D_out),
          torch.nn.Softmax()
        ).to(device)

modelOne = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, H-5),
          torch.nn.ReLU(),
          torch.nn.Linear(H-5, H-10),
          torch.nn.ReLU(),
          torch.nn.Linear(H-10, D_out),
          torch.nn.Sigmoid()
        ).to(device)"""

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
loss_fn = torch.nn.BCELoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
t1 = time.time()
for t in range(50):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.

  permutation = torch.randperm(T)
  for i in range(0,T, N):
      optimizer.zero_grad()

      indices = permutation[i:i+N]
      batch_x, batch_y = x[indices], y[indices]

        # in case you wanted a semi-full example
      batch_y_pred = model.forward(batch_x)
      loss = loss_fn(batch_y_pred,batch_y)


      model.zero_grad()

      loss.backward()
      optimizer.step()
      with torch.no_grad():
        for param in model.parameters():
          param.data -= learning_rate * param.grad
  print(t, loss.item())

t2 = time.time()
print(t2-t1)

#Check validation numbers
x_list = []
for i in range(0,10000):
  x_list.append(torch.load(os.path.join('val', str(i)+'.pt')))

x = torch.stack(x_list)
y = torch.index_select(x, 1, torch.tensor([4]))
y = toSoftmax(y)
x = torch.index_select(x, 1, torch.tensor(range(4)))

y_pred = model(x)

# Compute and print loss. We pass Tensors containing the predicted and true
# values of y, and the loss function returns a Tensor containing the loss.

"""incorrectGuesses = 0
for i in range(0, 10000):
  guess = [0, y_pred[i][0]]
  for j in range(3):
    if y_pred[i][j+1] > guess[1]:
      guess = [j, y_pred[i][j]]
  if not guess[0] == y[i]:
    incorrectGuesses += 1
print(incorrectGuesses)"""
print(torch.sum(torch.abs(torch.round(y_pred)-y)))

loss = loss_fn(y_pred, y)
print("validation loss: ", loss.item())
'''for i in range(5):
  print(model(x[i]) , "  ", y[i])'''

