#Step 1: load and normalize CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms
#Can download data sets by changing download in the trainset, and testset initilization to True
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = False, transform = transform)

testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                        download = False, transform = transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                          shuffle = True, num_workers = 2)

testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
                                          shuffle = False, num_workers = 2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#examine some of the training images
'''
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + .05
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg (1, 2, 0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()

print('  '.join('%5s' % classes[labels[j]] for j in range(4)))
'''
#step 2 build the neural net
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

net = Net()

#step 3. Define the Loss function
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = .001, momentum = .09)

#Step 4 Train the neural Network

for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterions(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
