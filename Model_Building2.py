import torch, torch.nn
import numpy as np
from scipy.sparse import coo_matrix
import Preprocessor2
import time

def ModelBuilding(decsize, epochs):
    st = time.time()
    x = r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSLUntamped\KDDTrain.csv'
    y = r"C:\Users\vdevk\OneDrive\Desktop\GridSearch\KDD_NSLUntamped\KDDTest.csv"

    train_normalonly, test_malwareonly, test_normalonly, test_malwareonly1, avgperentry = Preprocessor2.Preprocessor.initial(x, y)
    sysfeat = train_normalonly.values.tolist()
    test_normalonly = test_normalonly.values.tolist()
    test_malwareonly = test_malwareonly.values.tolist()


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
                torch.nn.Linear(121, decsize),
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
                torch.nn.Linear(decsize, 121)
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

            # Print the loss every Epoch
            i+=1
            losssum = losssum + loss
            if i%len(sysfeat) == 0:
                losses.append(losssum.detach().numpy()/len(sysfeat))
                print("Epoch " + str(epoch+1) + " Loss: " + str(losssum/len(sysfeat) * 1000))
                losssum = 0

    et = time.time()
    elapsed = et-st
    print("Autoencoder Done")
    #  Save model
    torch.save(model.state_dict(), 'systrep.sav')
    print("Autoencoder Saved", elapsed)


    #  Initialize both models w/ parameters trained from main.py and PhaseII.py
    model1 = Sysrep()
    model1.load_state_dict(torch.load('systrep.sav'))
    model1.eval()

    #  Prepare a Sparse Matrix
    row  = []
    col  = []
    data = []
    print(np.shape(test_normalonly))
    for i in range(len(test_normalonly)):
        for j in range(121):
            if test_normalonly[i][j] != 0:
                row.append(i)
                col.append(j)
                data.append(test_normalonly[i][j])
    test = coo_matrix((data, (row, col)), shape=(len(test_normalonly), 121))

    values = test.data
    indices = np.vstack((test.row, test.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = test.shape
    test = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    #  Generate predictions of the testing data, check if the predictions are correct
    phase1 = []
    prediction = []

    for data in range(len(test)):
        phase1.append(model1.pretrain(test[data]).detach().numpy())
    phase1 = torch.FloatTensor(np.array(phase1))
    test_normalonly = np.delete(test_normalonly, -1, axis=1)

    msetestnorm = ((phase1 - test_normalonly)**2).mean(axis=1)
    msetestnorm = msetestnorm.cpu().detach().numpy()
    print(np.mean(msetestnorm))


    #  Prepare a Sparse Matrix
    row  = []
    col  = []
    data = []
    for i in range(len(test_malwareonly)):
        for j in range(121):
            if test_malwareonly[i][j] != 0:
                row.append(i)
                col.append(j)
                data.append(test_malwareonly[i][j])
    test = coo_matrix((data, (row, col)), shape=(len(test_malwareonly), 121))

    values = test.data
    indices = np.vstack((test.row, test.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = test.shape
    test = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    #  Generate predictions of the testing data, check if the predictions are correct
    phase1 = []
    prediction = []

    for data in range(len(test)):
        phase1.append(model1.pretrain(test[data]).detach().numpy())
    phase1 = torch.FloatTensor(np.array(phase1))
    test_malwareonly = np.delete(test_malwareonly, -1, axis=1)

    msetestmal = ((phase1 - test_malwareonly)**2).mean(axis=1)
    msetestmal = msetestmal.cpu().detach().numpy()
    print(np.mean(msetestmal))


decsize = [1]
epochs = [10]
for i in range(5):
    ModelBuilding(decsize[0], epochs[0])












