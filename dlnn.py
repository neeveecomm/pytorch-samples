import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())

for x in range(len(params)):
    print('Param Index: {} Params : {}'.format(x, params[x].size()))


#learning_rate = 0.01
#for f in net.parameters():
#    f.data.backward()
#    f.data.sub_(f.grad.data * learning_rate)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

target = torch.randn(10)
target = target.view(1, -1)
crit = nn.MSELoss()
loss = crit(out, target)
print(loss)

net.zero_grad()
print('bias before backward {}'.format(net.conv1.bias.grad))
loss.backward()
optimizer.step()
print('bias after backward {}'.format(net.conv1.bias.grad))