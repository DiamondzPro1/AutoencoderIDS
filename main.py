import pandas
import torch, torch.nn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import random


#  Load Data, Perform OneHot Normalization
df = pandas.read_csv(r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSL.csv')
df = df.dropna(axis=1, how= 'all')


trans1 = make_column_transformer((OneHotEncoder(), ['Column2', 'Column3', 'Column4']), remainder = 'passthrough')
df = trans1.fit_transform(df)

scaler = MaxAbsScaler()
df = scaler.fit_transform(df)

#  Load df into unsplit array
unsplit = []
for row in df:
    unsplit.append(row)
#  Create the unlearned features dataset
b1 = 100000
b2= 130000
typefeat = []
sysfeat = []
size = 100000
for i in range(size):
    sysfeat.append(unsplit[i+1])
print(type(sysfeat), np.shape(sysfeat))

#  Prepare a Sparse Coo Matrix
row  = []
col  = []
data = []
for i in range(len(sysfeat)):
    for j in range(122):
        if sysfeat[i][j].any() != 0:
            row.append(i)
            col.append(j)
            data.append(sysfeat[i][j])

sysfeat = coo_matrix((data, (row, col)), shape=(size, 122))

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
            torch.nn.Linear(122, 61),
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
            torch.nn.Linear(61, 122),
            torch.nn.LeakyReLU()
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
    random.shuffle(sysfeat)
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
        if i%size == 0:
            losses.append(losssum.detach().numpy()/size)
            print("Epoch " + str(epoch+1) + " Loss: " + str(losssum/size))
            losssum = 0
        #if i%100 == 0:
           # print("Epoch " + str(epoch+1) + " Point: " + str(i))
            #print(model.features(point))

#  Save model
torch.save(model.state_dict(), 'systrep.sav')
for name, param in model.named_parameters():
    print(name, param.data)
print("Autoencoder Saved")
print(model)

plt.plot(losses)
plt.show()