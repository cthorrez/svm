import numpy as np 
import copy

def load_data():
	X = np.genfromtxt('../DATA/attributes.csv', delimiter=',', dtype=float)
	y = np.genfromtxt('../DATA/labels.csv', delimiter=',')
	return X, y



def compute_cost(X, Y, w, C, b):
	weight_cost = 0.5*np.dot(w,w)
	other_cost = 0
	for x,y in zip(X,Y):
		other_cost += C*max(0, 1-y*(np.dot(w,x) + b))
	return weight_cost + other_cost


def get_j_grad(x,y,w,b,j,C):
	gamma = 0
	if y*(np.dot(x,w)+b) < 1:
		gamma = -1*y*x[j]
	return w[j] + C*gamma




def SGD(X, Y, lr, epsilon, C):
	ks = []
	costs = []

	dim = X.shape[1]
	n = X.shape[0]
	w = np.zeros(dim)
	b = 0.0
	delta_cost = 0.0
	delta_cost_p = 0.0
	i = 1
	k = 0
	cost = compute_cost(X, Y, w, C, b)
	print cost
	delta_cost = cost
	while delta_cost > epsilon:
		ks.append(k)
		costs.append(cost)

		x = X[i]
		y = Y[i]
		w_old = copy.deepcopy(w)
		for j in range(dim):
			grad = get_j_grad(x,y,w_old,b,j,C)
			w[j] = w_old[j]-lr*grad
		delta2 = 0
		if y*(np.dot(x,w_old)+b) < 1:
			delta2 = -1*y
		b = b - lr*C*delta2

		k = k + 1
		i = (i+1)%n

		old_cost = cost
		cost = compute_cost(X, Y, w, C, b)
		new_cost = cost
		
		old_delta_cost = delta_cost
		delta_cost_p = 100*np.abs(old_cost-new_cost)/old_cost
		delta_cost = 0.5*old_delta_cost + 0.5*delta_cost_p


		print delta_cost
	return ks, costs



def main():
	X, Y = load_data()

	np.random.seed(498)
	perm = np.random.permutation(X.shape[0])
	X = X[perm]
	Y = Y[perm]

	ks, costs = SGD(X, Y, lr=1e-4, epsilon=1e-3, C=100)
	outfile = open('SGD_cost.csv', 'w')
	for k,cost in zip(ks,costs):
		outfile.write(','.join([str(k), str(cost)]) + '\n')
		outfile.flush()
	outfile.close()






if __name__ == '__main__':
	main()