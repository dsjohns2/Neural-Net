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
		self.num_input_params = 9
		self.num_classes = 3
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
test_data = np.loadtxt("test_data.txt")

# Train the net
net = Net()
num_epochs = 1000
batch_size = 4
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.001)
loss_arr = []
for epoch in range(0, num_epochs):
	average_loss = 0
	print("Current epoch: " + str(epoch))
	for i in range(0, len(train_data), batch_size):
		X = train_data[i:i+batch_size, 0:net.num_input_params]
		X = X.astype(np.float32)
		X = torch.from_numpy(X)
		y_real = train_data[i:i+batch_size, net.num_input_params]
		y_real = y_real.astype(np.int_)
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
	

# Test the net
num_correct = 0
num_wrong = 0
guess_count = np.zeros(net.num_classes)
for i in range(0, len(test_data)):
	X = test_data[i, 0:net.num_input_params]
	X = X.astype(np.float32)
	X = torch.from_numpy(X)
	y_real = test_data[i, net.num_input_params]
	y_guess = net(X)
	y_guess = np.argmax(y_guess.detach().numpy())
	guess_count[y_guess] += 1
	print("X: " + str(X.numpy()) + ", y_real: " + str(y_real) + ", y_net_guess: " + str(y_guess))
	if(y_guess == y_real):
		num_correct += 1
	else:
		num_wrong += 1

print("Percentage guessed correctly: " + str(num_correct/(num_correct+num_wrong)))
print("Net guess count: " + str(guess_count))

# Plot loss
loss_arr = np.asarray(loss_arr)
plt.plot(loss_arr[:, 0], loss_arr[:, 1])
plt.savefig("loss.png")
