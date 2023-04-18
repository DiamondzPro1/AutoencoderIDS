import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, LabelEncoder
from more_itertools import locate

class Preprocessor():
    def initial(x, y):
    #  Import the train and validation datasets. Train set is for the autoencoder, validation set is for the FNN.
        dftrain = pd.read_csv(x)
        dftest = pd.read_csv(y)

        #  Drop the 'nan' columns that might be present and Column 20 which has all 0's
        dftrain = dftrain.dropna(axis=1, how='all')
        dftrain = dftrain.drop(['Column20', 'Column43'], axis=1)

        dftest = dftest.dropna(axis=1, how='all')
        dftest = dftest.drop(['Column20', 'Column43'], axis=1)
        dftrainlen = len(dftrain)
        dfconcat = pd.concat([dftest, dftrain])

        #  OneHotEncode and Scale each dataframe, preserve datatypes of both objects
        trans1 = make_column_transformer((OneHotEncoder(), ['Column2', 'Column3', 'Column4']), remainder='passthrough')
        trans2 = MaxAbsScaler()

        dfconcat = pd.DataFrame(trans1.fit_transform(dfconcat))
        set_value = lambda x: 1 if x == 'normal' else 0
        dfconcat[121] = dfconcat[121].apply(set_value)
        dfconcat = pd.DataFrame(trans2.fit_transform(dfconcat))

        dftrain = dfconcat[0:dftrainlen]
        dftest = dfconcat[dftrainlen:len(dfconcat)]

        train_normalonly = dftrain[dftrain[121] != 0]
        train_malwareonly = dftrain[dftrain[121] == 0]
        test_normalonly = dftest[dftest[121] != 0]
        test_malwareonly = dftest[dftest[121] != 0]

        mean = train_normalonly.mean(axis = 1)

        return(train_normalonly, train_malwareonly, test_normalonly, test_malwareonly, mean)




