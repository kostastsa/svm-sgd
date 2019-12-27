import numpy as np
import matplotlib.pyplot as plt
import csv
import os
# Data pre-processing into numpy array. We want to classify zeros vs non-zeros
# and we assign value +1 to zeros, -1 to all other digits. There are 60000 digits
# in the training dataset of which approximatelly 6000 are zeros.

os.chdir("/Users/kostastsampourakis/Desktop/SVM project")

pre_data = []
with open("mnist_train.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = "\n")
    for row in reader:
        pre_data.append(row[0].split(","))

train_data = []
for row in pre_data:
    row = [int(i) for i in row]
    train_data.append(row)

train_data = np.array(train_data)

for row in train_data:
    if row[0] == 0 : row[0] = 1
    else : row[0] = -1

# Homogenization
Y = train_data[:,0]
X = train_data[:,1:]
X_hom = np.insert(X, 0, 1, axis=1)


# Test Data
pre_data2 = []
with open("mnist_test.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = "\n")
    for row in reader:
        pre_data2.append(row[0].split(","))

test_data = []
for row in pre_data2:
    row = [int(i) for i in row]
    test_data.append(row)

test_data = np.array(test_data)

for row in test_data:
    if row[0] == 0 : row[0] = 1
    else : row[0] = -1

# Homogenization
Y_t = test_data[:,0]
X_t = test_data[:,1:]
X_ht = np.insert(X_t, 0, 1, axis=1)
m_test = len(X_t)
nZero = sum(Y_t+1)/2


# Stochastic Gradient Descent

m = len(X)  # number of training examples
T = 10000  # number of rounds
d = 28*28
nsim = 1000

mstk_sim = []
for sim in range(nsim):
    print(sim)
    th_old = np.zeros(d+1)
    l = 1
    w = np.empty([T,d+1])
    th_new = 0
    for t in range(T):
        w[t] = th_old / (l*(t+1))
        i = np.random.randint(0,m)
        if Y[i]*np.dot(w[t], X_hom[i])<1:
            th_new = th_old + Y[i]*X_hom[i]
        else:
            th_new = th_old
        th_old = th_new


    w_bar = sum(w)/T


    # Generalization error

    Y_pred = []
    for i in range(m_test):
        if np.dot(w_bar, X_ht[i])>0:
            Y_pred.append(1)
        else:
            Y_pred.append(-1)
    Y_pred = np.array(Y_pred)

    mstk = np.linalg.norm(Y_pred-Y_t)**2/4
    nZero_p = sum(Y_pred+1)/2
    mstk_sim.append(mstk)
    print(mstk)
#print("the number of predicted zeros is {} whereas the true number of zeros is {}".format(nZero_p,nZero))
#print("The prercentile error is {}%".format(100.0*abs(nZero_p - nZero)/nZero))
#print("The number of mistakes on the test dataset is {}".format(mstk))
#print("The misclassification error is {}".format(mstk/m_test))


with open("mstks.csv","w") as f:
    for mst in mstk_sim:
        f.write("{}\n".format(mst))
