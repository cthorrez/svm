import numpy as np


def get_data():
    X = np.genfromtxt(fname="../DATA/attributes.csv", delimiter=',', dtype=float)
    Y = np.genfromtxt(fname="../DATA/labels.csv", delimiter=",")
    return X, Y

# cost function
def compute_cost(w, b, attr, label, C):
    sum = 0
    for i in range (0, attr.shape[0]):
        sum += max(0, 1 - label[i]*(np.dot(w, attr[i]) + b))
    return 0.5*np.dot(w, w) + C*sum

# update percent cost change
def compute_delta_cost(cost_old, cost_new):
    return 100.0*np.abs(cost_old - cost_new)/(cost_old)

# gradient wrt w
def gradient_w(w, attr, label, C, j, b):
    alpha = 0
    for i in range(0, label.shape[0]):
        if label[i]*(np.dot(attr[i], w) + b) < 1:
            alpha -= label[i]*attr[i][j]
    return w[j] + C*alpha

# gradient wrt b
def gradient_b(w, attr, label, b, C):
    beta = 0
    for i in range(0, label.shape[0]):
        if label[i] * (np.dot(attr[i], w) + b) < 1:
            beta -= label[i]
    return C*beta

# batch gradient descent
def main():
    attr, label = get_data()
    # initialize
    w = np.zeros(attr.shape[1])
    b = 0
    k = 0
    learning_rate = 0.3e-6
    epslion = 0.25
    delta_cost_p = epslion + 0.01
    C = 100
    cost_old = compute_cost(w, b, attr, label, C)
    cost_new = cost_old
    costs = [cost_old]
    iters = [0]
    while delta_cost_p >= epslion:
        w_old = w.copy()
        cost_old = cost_new
        for j in range(0, attr.shape[1]):
            w[j] = w_old[j] - learning_rate*gradient_w(w_old, attr, label, C, j, b)
        b = b - learning_rate*gradient_b(w, attr, label, b, C)
        k = k + 1
        cost_new = compute_cost(w, b, attr, label, C)
        print k, cost_new
        costs.append(cost_new)
        iters.append(k)
        delta_cost_p = compute_delta_cost(cost_old, cost_new)
    write_output(iters, costs)

def write_output(iters, costs):
    f = open("batch_cost.csv", "w")
    for i in range(0, len(iters)):
        f.write(str(iters[i]) + "," + str(costs[i]) + "\n")
    f.close()

if __name__ == "__main__":
    main()