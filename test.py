'''
import torch
import numpy as np

# examples
# create tensors of different dimensions with different data types
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype = torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

# create tensors from existing tensors

x = x.new_ones(5, 3, dtype = torch.double)
print(x)

x = torch.randn_like(x, dtype = torch.float)
print(x)


# get size of tensor
print(x.size())

# tensor operations
y = torch.rand(5, 3)

print(x + y)
print(torch.add(x, y))

#reshape tensors

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# one element tensors have the item method to retrieve

x = torch.randn(1)
print(x)
print(x.item())


#using tensors on the gpu
if torch.cuda.is_available():
    device = torch.device("cuda")           # a cuda device object
    y = torch.ones_like(x, device = device) # directly create a tensor on gpu
    x = x.to(device)                        # or just use strings ...to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
    print("dub")


#using numpy with torch
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print(a)
print(b)
'''

#creating a neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=  s
        return num_features

net = Net()
print(net)

# the learnable parameters of a model are returned by ''net.parameters()''
params = list(net.parameters())
print(len(params))
print(params[0].size()) #conv1's .weight

# try a random 32*32 input
#expected input size to this net is 32 x 32
#Mnist dataset, resize the images from the dataset to 32*32

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


#zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.rand(1, 10))


#loss function example
output = net(input)
target = torch.randn(10)
target = target.view(1, -1) # make this the same shape as the output of the network
criterion = nn.MSELoss()
loss = criterion(output, target)
print('Loss', loss)


#For illustration of the loss function
print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0][0]) # linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #relu

#backwards propagation

net.zero_grad()     #zeroes the gradien buffers of all parameters

print('conv1.bias.grad before backwards')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad before backwards')
print(net.conv1.bias.grad)

#actually apply the SGD or Update the Weights
import torch.optim as optim

#create an optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

#in your training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
