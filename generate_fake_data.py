import numpy as np
import random

def generate_data():
	data = []
	for i in range(0, 1000):
		x1 = random.random()
		x2 = random.random()
		x3 = random.random()
		x4 = random.random()
		x5 = random.random()
		x6 = random.random()
		x7 = random.random()
		x8 = random.random()
		x9 = random.random()
		sum = x1 + x5 + x8
		if(sum > 2):
			y = 2
		elif(sum > 1):
			y = 1
		else:
			y = 0
		data_point = [x1, x2, x3, x4, x5, x6, x7, x8, x9, y]
		data.append(data_point)

	data = np.asarray(data)
	return data

np.savetxt("train_data.txt", generate_data())
np.savetxt("test_data.txt", generate_data())
