import pandas
import torch, torch.nn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import random
import Preprocessor1
import tensorflow as tf


x = r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSLUntamped\KDDTrain.csv'
y = r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSLUntamped\KDDTestV.csv'
z = r"C:\Users\vdevk\OneDrive\Desktop\GridSearch\KDD_NSLUntamped\KDDTest.csv"

sysfeat, attfeat, c, atttype, e, f = Preprocessor1.Preprocessor.initial(x, y, z)
sysfeat = sysfeat.values.tolist()
attfeat = attfeat.values.tolist()
atttype = atttype.values.tolist()

row  = []
col  = []
data = []
for i in range(len(sysfeat)):
    for j in range(121):
        if sysfeat[i][j] != 0:
            row.append(i)
            col.append(j)
            data.append(sysfeat[i][j])

sysfeat = coo_matrix((data, (row, col)), shape=(len(sysfeat), 121))

#  Transform to Tensor
values = sysfeat.data
indices = np.vstack((sysfeat.row, sysfeat.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = sysfeat.shape
sysfeat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

#  Use an Autoencoder to select 61/122 prevalent features from sysfeat. Should I condense the data further?
class Sysrep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(121, 61),
            torch.nn.LeakyReLU(),
            #torch.nn.Linear(110, 100),
            #torch.nn.Dropout(.15),
            #torch.nn.ReLU(),
            #torch.nn.Linear(30, 15),
            #torch.nn.Dropout(.1),
            #torch.nn.Sigmoid(),
            #torch.nn.Linear(15, 7),

        )
        self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(7, 15),
            #torch.nn.Dropout(.1),
            #torch.nn.ReLU(),
            #torch.nn.Linear(15, 30),
            #torch.nn.Dropout(.1),
            #torch.nn.Sigmoid(),
            #torch.nn.Linear(100, 110),
            #torch.nn.Dropout(.05),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(61, 121)
            #torch.nn.LeakyReLU()
        )
    def pretrain(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def features(self, x):
        encoded = self.encoder(x)
        return encoded

#  Model Initialization
model = Sysrep()
#  Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
#  Using an Adam Optimizer with lr = 1e-6 and wd = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0)

#  Training
epochs = 1
outputs = []
losses = []
losssum = 0
for epoch in range(epochs):
    i = 0
    #random.shuffle(sysfeat)
    for point in sysfeat:
        # Output of Autoencoder
        reconstructed = model.pretrain(point)
        # Calculating the loss function
        loss = loss_function(reconstructed, point)
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print the loss every 100th point

        i+=1
        losssum = losssum + loss
        if i%len(sysfeat) == 0:
            losses.append(losssum.detach().numpy()/len(sysfeat))
            print("Epoch " + str(epoch+1) + " Loss: " + str(losssum/len(sysfeat)))
            losssum = 0
        #if i%100 == 0:
           # print("Epoch " + str(epoch+1) + " Point: " + str(i))
            #print(model.features(point))

print("Autoencoder Done")
#  Save model
torch.save(model.state_dict(), 'systrep.sav')
for name, param in model.named_parameters():
    print(name, param.data)
print("Autoencoder Saved")

#  Shuffle the input and output accordingly
unrandom = list(zip(attfeat, atttype))
random.shuffle(unrandom)
attfeat, atttype = zip(*unrandom)

#  Convert the dataframes to tensors
row  = []
col  = []
data = []
for i in range(len(attfeat)):
    for j in range(121):
        if attfeat[i][j] != 0:
            row.append(i)
            col.append(j)
            data.append(attfeat[i][j])
attfeat = coo_matrix((data, (row, col)), shape=(len(attfeat),121))

values = attfeat.data
indices = np.vstack((attfeat.row, attfeat.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = attfeat.shape
attfeat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
atttype = torch.tensor(atttype)

#  Generate feature representation of 675 points to use in categorization w/ autoencoder saved from main.py
model1 = Sysrep()
phase1 = []
for data in range(len(attfeat)):
    phase1.append(model1.features(attfeat[data]).detach().numpy())

#  Convert the phase1 tensor and atttype tensor into appropriate dtypes for CrossEntropyLoss()
phase1 = torch.FloatTensor(np.array(phase1))
atttype = torch.LongTensor(np.array(atttype))

print("FNN Data Processing Done")
#  Create Categorization model. Should I deepen the hidden layer? I found 750 nodes provides optimal training at ~80% accuracy
class Logits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Logit = torch.nn.Sequential(
            torch.nn.Linear(61, 59),
            torch.nn.ReLU(),
            torch.nn.Linear(59, 57),
            torch.nn.ReLU(),
            torch.nn.Linear(57, 55),
            torch.nn.ReLU(),
            torch.nn.Linear(55, 53),
            torch.nn.ReLU(),
            torch.nn.Linear(53, 51),
            torch.nn.ReLU(),
            torch.nn.Linear(51, 49),
            torch.nn.ReLU(),
            torch.nn.Linear(49, 47),
            torch.nn.ReLU(),
            torch.nn.Linear(47, 45),
            torch.nn.ReLU(),
            torch.nn.Linear(45, 43),
            torch.nn.ReLU(),
            torch.nn.Linear(43, 40)
            #torch.nn.ReLU(),
            #torch.nn.Linear(600, 500),
            #torch.nn.ReLU(),
            #torch.nn.Linear(500, 400),
            #torch.nn.ReLU(),
            #torch.nn.Linear(400, 300),
            #torch.nn.ReLU(),
            #torch.nn.Linear(300, 200),
            #torch.nn.ReLU(),
            #torch.nn.Linear(200, 100),
            #torch.nn.ReLU(),
            #torch.nn.Linear(100, 40)
        )
    def pretrain(self, x):
        logits = self.Logit(x)
        return logits


#  Initialize Logits() and loss function
model2 = Logits()
loss_function = torch.nn.CrossEntropyLoss()
for param in model2.parameters():
    param.requires_grad = True

#  Using an SGD Optimizer with lr = 0.001 and wd = 1e-4
optimizer = torch.optim.Adam(model2.parameters(), lr = .01, weight_decay = 0)

#  Training
epochs = 200
outputs = []
losses = []
losssum = 0
correct = 0

for epoch in range(epochs):
    i = 0
    #if epoch%100 == 0:
    unrandom = list(zip(phase1, atttype))
    random.shuffle(unrandom)
    phase1, atttype = zip(*unrandom)
    atttype = torch.LongTensor(atttype)

    for data in range(len(phase1)):
        # Output of Categorization class
        phase1[data].requires_grad = True
        reconstructed = model2.pretrain(phase1[data])

        #  Reformat the ground truth and predicted tensors for loss function
        ypred = torch.FloatTensor(reconstructed)
        ypred = ypred.view(1, 40)
        #print(ypred)
        atttype = atttype.view(len(attfeat),1)
        regularization_loss = 0
        for param in model2.parameters():
            regularization_loss += torch.sum(torch.abs(param))
        #print(regularization_loss)
        #print(loss_function(ypred, atttype[data]))

        # Calculating the loss function
        loss = loss_function(ypred, atttype[data]) + (regularization_loss) * 0
        losssum = losssum + loss.item()

        #  Store previous 1000 point loss average for plotting


        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classification = torch.argmax(ypred)

        if classification == atttype[data]:
            correct += 1

        i += 1
    losses.append(losssum / len(attfeat))
    print("Epoch " + str(epoch + 1) + " Loss: " + str(losssum / len(attfeat)) + " Accuracy: " + str(correct))
    losssum = 0
    correct = 0

for name, param in model2.named_parameters():
    print(name, param.data)
#  Save Logits() model, plot loss graph
torch.save(model2.state_dict(), 'classifier.sav')
print("Model Saved")
plt.plot(losses)
plt.show()






