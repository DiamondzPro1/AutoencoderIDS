import pandas
import torch, torch.nn
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, LabelEncoder
import numpy as np
from scipy.sparse import coo_matrix
import sklearn.metrics


#  Load in the 2 models from main.py and PhaseII.py
class Sysrep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(122, 61),
            torch.nn.ReLU(),
           # torch.nn.Linear(61, 30),
            #torch.nn.Dropout(.15),
            #torch.nn.ReLU(),
            #torch.nn.Linear(30, 15),
            #torch.nn.Dropout(.1),
            #torch.nn.ReLU(),
            #torch.nn.Linear(15, 7),

        )
        self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(7, 15),
            #torch.nn.Dropout(.1),
            #torch.nn.ReLU(),
            #torch.nn.Linear(15, 30),
            #torch.nn.Dropout(.1),
            #torch.nn.ReLU(),
            #torch.nn.Linear(30, 61),
            #torch.nn.Dropout(.05),
            #torch.nn.ReLU(),
            torch.nn.Linear(61, 122),
            #torch.nn.Sigmoid()
        )

    def pretrain(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def features(self, x):
        encoded = self.encoder(x)
        return encoded


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


#  Initialize both models w/ parameters trained from main.py and PhaseII.py
model1 = Sysrep()
model1.load_state_dict(torch.load('systrep.sav'))
model1.eval()
model2 = Logits()
model2.load_state_dict(torch.load('classifier.sav'))
model2.eval()
#for name, param in model2.named_parameters():
    #print(name, param.data)

#  Load Data, Perform OneHot, Normalization, and LabelEncoding on Classes
dfin = pandas.read_csv(r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSL.csv')
trans1 = make_column_transformer((OneHotEncoder(), ['Column2', 'Column3', 'Column4']), remainder = 'passthrough')
dfin = trans1.fit_transform(dfin)
scaler = MaxAbsScaler()
dfin = scaler.fit_transform(dfin)
unsplit1 = []
test = []
for row in dfin:
    unsplit1.append(row)
test = unsplit1[130000:]

dfout = pandas.read_csv(r'C:\Users\vdevk\OneDrive\Desktop\Datasets\KDD_NSLLabels.csv')
unsplit = []
gt = []
unsplit2 = dfout.to_numpy()
gt = unsplit2[130000:]
le = LabelEncoder()
fit = le.fit(gt)
gt = le.transform(gt)

#  Prepare a Sparse Matrix
row  = []
col  = []
data = []
for i in range(len(test)):
    for j in range(122):
        if test[i][j].any() != 0:
            row.append(i)
            col.append(j)
            data.append(test[i][j])
test = coo_matrix((data, (row, col)), shape=(30367, 122))

values = test.data
indices = np.vstack((test.row, test.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = test.shape
test = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
atttype = torch.tensor(gt)

#  Generate predictions of the testing data, check if the predictions are correct
phase1 = []
prediction = []
correct = 0
atttype = torch.LongTensor(np.array(atttype))
pred = []

for data in range(len(test)):
    phase1.append(model1.features(test[data]).detach().numpy())
phase1 = torch.FloatTensor(np.array(phase1))
gt = np.array(gt)
for data in range(len(test)):
    prediction = model2.pretrain(phase1[data])
    ypred = torch.FloatTensor(prediction)
    ypred = ypred.view(1, 40)
    #print(ypred)
    atttype = atttype.view(30367, 1)
    classification = torch.argmax(prediction)
    #print(classification, gt[data])
    if classification.item() == gt[data]:
        correct +=1
    pred.append(classification.item())

np.set_printoptions(threshold = np.inf)
matrix = sklearn.metrics.confusion_matrix(gt, pred)
print("Test Accuracy: " + str(correct))
print(matrix)
#for i in range(len(matrix)):
    #print("Normal Predictions: " + str((matrix[i][16])))

intertruth = []
interpredict = []
for i in range(len(gt)):
    if gt[i] == 16:
        intertruth.append(0)
    elif gt[i] == 1 or gt[i] == 8 or gt[i] == 14 or gt[i] == 19 or gt[i] == 27 or gt[i] == 31 or gt[i] == 10 or gt[i] == 21 or gt[i] == 32 or gt[i] == 0 or gt[i] == 34:
        intertruth.append(1)
    elif gt[i] == 25 or gt[i] == 7 or gt[i] == 15 or gt[i] == 20 or gt[i] == 11 or gt[i] == 24:
        intertruth.append(2)
    elif gt[i] == 4 or gt[i] == 3 or gt[i] == 6 or gt[i] == 18 or gt[i] == 12 or gt[i] == 33 or gt[i] == 35 or gt[i] == 36 or gt[i] == 29 or gt[i] == 28 or gt[i] == 26 or gt[i] == 13 or gt[i] == 5:
        intertruth.append(3)
    else:
        intertruth.append(4)

for i in range(len(pred)):
    if pred[i]==16:
        interpredict.append(0)
    elif pred[i] == 1 or pred[i] == 8 or pred[i] == 14 or pred[i] == 19 or pred[i] == 27 or pred[i] == 31 or pred[i] == 10 or pred[i] == 21 or pred[i] == 32 or pred[i] == 0 or pred[i] == 34:
        interpredict.append(1)
    elif pred[i] == 25 or pred[i] == 7 or pred[i] == 15 or pred[i] == 20 or pred[i] == 11 or pred[i] == 24:
        interpredict.append(2)
    elif pred[i] == 4 or pred[i] == 3 or pred[i] == 6 or pred[i] == 18 or pred[i] == 12 or pred[i] == 33 or pred[i] == 35 or pred[i] == 36 or pred[i] == 29 or pred[i] == 28 or pred[i] == 26 or pred[i] == 13 or pred[i] == 5:
        interpredict.append(3)
    else:
        interpredict.append(4)

intermat = sklearn.metrics.confusion_matrix(intertruth, interpredict)

print(intermat)

finaltruth = []
finalpred = []
for i in range(len(gt)):
    if intertruth[i] == 0:
        finaltruth.append(0)
    else:
        finaltruth.append(1)

for i in range(len(gt)):
    if interpredict[i] == 0:
        finalpred.append(0)
    else:
        finalpred.append(1)

intermat = sklearn.metrics.confusion_matrix(finaltruth, finalpred)

print(intermat)
