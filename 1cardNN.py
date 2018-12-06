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

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# T is size of training set
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
T, N, D_in, H, D_out = 100000, 200, 5*4*13, 13, 4

# Create random Tensors to hold inputs and outputs
x_list = []
for i in range(0,100000):
  x_list.append(torch.load(os.path.join('trn', str(i)+'.pt')))

x = torch.stack(x_list)
y = torch.index_select(x, 1, torch.tensor([217]))
x = torch.index_select(x, 1, torch.tensor(range(217)))
#x = torch.rand(N, D_in, device=device)s
#y = torch.rand(N, D_out, device=device)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
          torch.nn.Conv3d(1, H, kernel_size = (3, 3, 3), stride=1, padding = (1,1,1)),
          torch.nn.ReLU(),
          torch.nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2)),
          torch.nn.Linear(D_in, D_out),
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
        ).to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function. Setting
# reduction='sum' means that we are computing the *sum* of squared errors rather
# than the mean; this is for consistency with the examples above where we
# manually compute the loss, but in practice it is more common to use mean
# squared error as a loss by setting reduction='elementwise_mean'.
loss_fn = torch.nn.BCELoss()

learning_rate = 1e-4
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
y = torch.index_select(x, 1, torch.tensor([217]))
x = torch.index_select(x, 1, torch.tensor(range(217)))

y_pred = model(x)

# Compute and print loss. We pass Tensors containing the predicted and true
# values of y, and the loss function returns a Tensor containing the loss.

print(torch.sum(torch.abs(torch.round(y_pred)-y)))

loss = loss_fn(y_pred, y)
print("validation loss: ", loss.item())
'''for i in range(5):
  print(model(x[i]) , "  ", y[i])'''