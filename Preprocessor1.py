import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, LabelEncoder

class Preprocessor():
    def initial(x, y, z):
    #  Import the train and validation datasets. Train set is for the autoencoder, validation set is for the FNN.
        dftrain = pd.read_csv(x)
        dfval = pd.read_csv(y)
        dftest = pd.read_csv(z)

    #  Drop the 'nan' columns that might be present and Column 20 which has all 0's
        dfin = dftrain.dropna(axis=1, how='all')
        dfin = dfin.drop('Column20', axis=1)
        dftest = dftest.dropna(axis=1, how='all')
        dftest = dftest.drop('Column20', axis=1)

    # Define a copy of the validation set and drop the potential 'nan' columns and Column 20. Convert this to a numpy array.
        dfval1 = dfval
        dfval1 = dfval1.dropna(axis=1, how='all')
        dfval1 = dfval1.drop('Column20', axis=1)
        dfval1 = pd.DataFrame.to_numpy(dfval1)

    # Concatinate both the dftrain and dfval sets and label encode the outputs. This ensures we have 40 classes from the
    # combination of the dftrain and dfval sets rather than the 38 that are in the dfval set. We test for 40 classes.
        dfconcat = pd.concat([dftrain, dfval, dftest])
        leout = dfval.iloc[:, -2]
        dfconcatout = dfconcat.iloc[:, -2]

    # Label Encode based off the 40 classes we are testing for.
        labencoder = LabelEncoder()
        fit = labencoder.fit(dfconcatout)
        classeslist = fit.classes_
        leout = labencoder.transform(leout)

    # Define a count array of each of the 40 classes. We have to select a limited amount of data from each class within the
    # validation set
        count = np.zeros(len(classeslist), dtype=int)

    # Append 10 datapoints from each class within the validation set (if present) within each of the 40 classes we test for.
    # Convert the result to a dataframe
        dfvalselect = []
        for data in range(len(dfval)):
            if count[leout[data]] < 2:
                dfvalselect.append(dfval1[data])
                count[leout[data]] +=1
        dfvalselect = pd.DataFrame(dfvalselect)

    # Rename the dfvalselect columns to the same names as dfin
        dfinnames = list(dfin.columns.values)
        dfvalselect.columns = dfinnames

    # Save the sizes of dfin and dfvalselect to split the dataset horizontally after the transformations
        autodimen = len(dfin)
        FNNdimen = len(dfvalselect)

    # Concat the dfin and dfvalselect sets to ensure no feature information is lost
        dftotal = pd.concat([dfin, dfvalselect, dftest])

    #  Split the dataset vertically to separate all inputs and outputs from one another
        dftotala = dftotal.iloc[:, :-2]
        dftotalb = pd.DataFrame(dftotal.iloc[:, -2])

    #  OneHotEncode and Scale each dataframe, preserve datatypes of both objects
        trans1 = make_column_transformer((OneHotEncoder(), ['Column2', 'Column3', 'Column4']), remainder='passthrough')
        trans2 = make_column_transformer((OneHotEncoder(), ['Column42']), remainder = 'passthrough')
        trans3 = MaxAbsScaler()

        dftotala = pd.DataFrame(trans1.fit_transform(dftotala))
        dftotalb = pd.DataFrame(labencoder.transform(dftotalb))
        dftotala = pd.DataFrame(trans3.fit_transform(dftotala))

    #  Split both objects into their Autoencoder and FNN pieces
        dfina = dftotala.loc[0:autodimen:1]
        dfvala = dftotala.loc[autodimen: autodimen + FNNdimen - 1:1]
        dftesta = dftotala.loc[autodimen + FNNdimen - 1: len(dftotala):1]
        dfinb = dftotalb.loc[0:autodimen:1]
        dfvalb = dftotalb.loc[autodimen:autodimen + FNNdimen - 1:1]
        dftestb = dftotalb.loc[autodimen + FNNdimen - 1: len(dftotalb):1]

        return(dfina, dfvala, dfinb, dfvalb, dftesta, dftestb)





