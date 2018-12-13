#Jake Williams and Samuel Tan

#Code for full double dummy solver neural network

import torch
import time
import os

def getValue(y):
  onehot = torch.zeros(y.size()[0])
  for i in range(y.size()[0]):
    onehot[i] = (int)(y[i][0][0][0].item())
  return onehot

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# T is size of training set
# N is batch size; D_in is input dimension(the number of 4x13 layers);
# H is hidden dimension; poolSize is the number of outputs from maxPooling; D_out is output dimension.
T, N, D_in, H, poolSize, D_out = 80000, 200, 4, 13, 26*4*13, 1

#load data
x_list = []
for i in range(0,T):
  x_list.append(torch.load(os.path.join('trn13', str(i)+'.pt')))
print("loaded training data")
x = torch.stack(x_list)
y = torch.index_select(x, 1, torch.tensor([4]))
y = getValue(y)
x = torch.index_select(x, 1, torch.tensor(range(4)))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(D_in, H, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(H, 2*H, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(poolSize, H)
        self.fc2 = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, poolSize)
        x = self.fc2(self.fc1(x))
        return 13*torch.sigmoid(x) #since output is now between 0 and 13

model = Net()

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
t1 = time.time()
for t in range(50):
  #generate batches

  permutation = torch.randperm(T)
  for i in range(0,T, N):
      optimizer.zero_grad()

      indices = permutation[i:i+N]
      batch_x, batch_y = x[indices], y[indices]

      batch_y_pred = model.forward(batch_x)
      loss = loss_fn(torch.squeeze(batch_y_pred),batch_y)


      model.zero_grad()

      loss.backward()
      optimizer.step()
      with torch.no_grad():
        for param in model.parameters():
          param.data -= learning_rate * param.grad
  print(t, loss.item())

t2 = time.time()
print(t2-t1)

#load validation data
x_list = []
for i in range(0,20000):
  x_list.append(torch.load(os.path.join('val13', str(i)+'.pt')))

x = torch.stack(x_list)
y = torch.index_select(x, 1, torch.tensor([4]))
y = getValue(y)
x = torch.index_select(x, 1, torch.tensor(range(4)))

y_pred = model(x)

# Compute and print loss. We pass Tensors containing the predicted and true
# values of y, and the loss function returns a Tensor containing the loss.

#to count the 0, 1, and 2 loss
correct = 0
y0 = 0
offByOne = 0
y1 = 0
offByTwo = 0
y2 = 0
for i in range(0, 20000):
  guess = torch.round(y_pred[i][0])
  actual = y[i]
  if y[i] == 0:
    y0 += 1
  elif y[i] == 1:
    y1 += 1
  else:
    y2 += 1
  if guess == actual:
    correct += 1
  elif torch.abs(guess - actual) == 1:
    offByOne += 1
  elif torch.abs(guess - actual) == 2:
    offByTwo += 1
print(correct)
print(offByOne)
print(offByTwo)


loss = loss_fn(torch.squeeze(y_pred), y)
print("validation loss: ", loss.item())


