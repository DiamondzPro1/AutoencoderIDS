import pandas
import torch, torch.nn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, LabelEncoder
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import random

#  Load Input DataFrame, Perform OneHot, Normalization, and LabelEncoding on Classes
dfin = pandas.read_csv(r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSL.csv')
dfin = dfin.dropna(axis='columns')
trans1 = make_column_transformer((OneHotEncoder(), ['Column2', 'Column3', 'Column4']), remainder = 'passthrough')
dfin = trans1.fit_transform(dfin)
scaler = MaxAbsScaler()
dfin = scaler.fit_transform(dfin)
unsplit1 = []
attfeat = []
for row in dfin:
    unsplit1.append(row)
for i in range(160366):
    attfeat.append(unsplit1[i])

#  Load Output DataFrame, Perform Label Encode to implement CrossEntropyLoss()
dfout = pandas.read_csv(r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSLLabels.csv')
unsplit2 = []
atttype = []
unsplit2 = dfout.to_numpy()
for i in range(160366):
    atttype.append(unsplit2[i])
le = LabelEncoder()
fit = le.fit(atttype)
atttype = le.transform(atttype)
for i in atttype:
    print(atttype[i])

#  Decrease the dataset size to prevent overfitting of the "normal" and "neptune" class
dummy = np.zeros(40, dtype=int)
selectedx = []
selectedy = []
for data in range(len(atttype)):
    if dummy[atttype[data]] < 100:
        selectedx.append(attfeat[data])
        selectedy.append(atttype[data])
        dummy[atttype[data]] +=1
attfeat = selectedx
atttype = selectedy
print(atttype)
print(len(atttype), len(attfeat), dummy)

#  Shuffle the input and output accordingly
unrandom = list(zip(selectedx, selectedy))
random.shuffle(unrandom)
attfeat, atttype = zip(*unrandom)

#  Convert the dataframes to tensors
row  = []
col  = []
data = []
for i in range(len(attfeat)):
    for j in range(122):
        if attfeat[i][j].any() != 0:
            row.append(i)
            col.append(j)
            data.append(attfeat[i][j])
attfeat = coo_matrix((data, (row, col)), shape=(2494,122))

values = attfeat.data
indices = np.vstack((attfeat.row, attfeat.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = attfeat.shape
attfeat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
atttype = torch.tensor(atttype)

#  Same model from main.py
class Sysrep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(122, 61),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(110, 100),
            # torch.nn.Dropout(.15),
            # torch.nn.ReLU(),
            # torch.nn.Linear(30, 15),
            # torch.nn.Dropout(.1),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(15, 7),

        )
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(7, 15),
            # torch.nn.Dropout(.1),
            # torch.nn.ReLU(),
            # torch.nn.Linear(15, 30),
            # torch.nn.Dropout(.1),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(100, 110),
            # torch.nn.Dropout(.05),
            # torch.nn.Sigmoid(),
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

#  Generate feature representation of 675 points to use in categorization w/ autoencoder saved from main.py
model1 = Sysrep()
model1.load_state_dict(torch.load('systrep.sav'))
phase1 = []
for data in range(len(attfeat)):
    phase1.append(model1.features(attfeat[data]).detach().numpy())

#  Convert the phase1 tensor and atttype tensor into appropriate dtypes for CrossEntropyLoss()
phase1 = torch.FloatTensor(np.array(phase1))
atttype = torch.LongTensor(np.array(atttype))

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
optimizer = torch.optim.Adam(model2.parameters(), lr = .0007, weight_decay = 0)

#  Training
epochs = 100
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
        atttype = atttype.view(2494,1)
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
    losses.append(losssum / 2494)
    print("Epoch " + str(epoch + 1) + " Loss: " + str(losssum / 2494) + " Accuracy: " + str(correct))
    losssum = 0
    correct = 0

for name, param in model2.named_parameters():
    print(name, param.data)
#  Save Logits() model, plot loss graph
torch.save(model2.state_dict(), 'classifier.sav')
print("Model Saved")
plt.plot(losses)
plt.show()

