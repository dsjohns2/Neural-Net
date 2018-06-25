import numpy as np
import random

def generate_data():
	data = []
	table_size = 7 
	table = np.zeros((table_size, table_size))
	for i in range(table_size):
		for j in range(table_size):
			table[i][j] = random.randint(0, 10)

	np.savetxt("table.txt", table)
	for i in range(0, 1000):
		x1 = random.randint(0, table_size-1)
		x2 = random.randint(0, table_size-1)
		y = table[x1][x2]
		data_point = [x1, x2, y]
		data.append(data_point)

	data = np.asarray(data)
	return data

np.savetxt("train_data.txt", generate_data())
