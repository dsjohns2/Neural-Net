from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tkinter
import matplotlib.pyplot as plt

# Neural Network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.num_input_params = 1
		self.num_classes = 1
		self.fc1 = nn.Linear(self.num_input_params, 120)
		self.fc2 = nn.Linear(120, 120)
		self.fc3 = nn.Linear(120, 60)
		self.fc4 = nn.Linear(60, self.num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)
		return x

# Get training and test data
train_data = np.loadtxt("train_data.txt")

# Train the net
net = Net()
num_epochs = 2000
batch_size = 40
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.001)
loss_arr = []
for epoch in range(0, num_epochs):
	average_loss = 0
	np.random.shuffle(train_data)
	print("Current epoch: " + str(epoch))
	for i in range(0, len(train_data), batch_size):
		X = train_data[i:i+batch_size, 0:net.num_input_params]
		X = X.astype(np.float32)
		X = torch.from_numpy(X)
		y_real = train_data[i:i+batch_size, net.num_input_params]
		y_real = y_real.reshape(batch_size, 1)
		y_real = y_real.astype(np.float32)
		y_real = torch.from_numpy(y_real)
		optimizer.zero_grad()
		y_guess = net(X)
		loss = criterion(y_guess, y_real)
		average_loss += float(loss)
		loss.backward()
		optimizer.step()
	average_loss /= (len(train_data)/batch_size)
	loss_arr.append([epoch, average_loss])
	print("Loss: " + str(average_loss))
	
# View Net
table_size = 10
for i in range(table_size):
	X = np.array([i])
	X = X.astype(np.float32)
	X = torch.from_numpy(X)
	print(net(X), end=' ')
print()

x_vals = np.arange(0, 10, .01)
y_vals = []
for x in x_vals:
	x = np.array([x])
	x = x.astype(np.float32)
	x = torch.from_numpy(x)
	y_vals.append(net(x))
plt.plot(x_vals, y_vals)
plt.savefig("1d_net.png")
