import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from matplotlib import style
from math import floor
import statistics
style.use("ggplot")

FEATURES = ["date","period","nswprice","nswdemand"]
# FEATURES =  ['DE Ratio',
#              'Trailing P/E',
#              'Price/Sales',
#              'Price/Book',
#              'Profit Margin',
#              'Operating Margin',
#              'Return on Assets',
#              'Return on Equity',
#              'Revenue Per Share',
#              'Market Cap',
#              'Enterprise Value',
#              'Forward P/E',
#              'PEG Ratio',
#              'Enterprise Value/Revenue',
#              'Enterprise Value/EBITDA',
#              'Revenue',
#              'Gross Profit',
#              'EBITDA',
#              'Net Income Avl to Common ',
#              'Diluted EPS',
#              'Earnings Growth',
#              'Revenue Growth',
#              'Total Cash',
#              'Total Cash Per Share',
#              'Total Debt',
#              'Current Ratio',
#              'Book Value Per Share',
#              'Cash Flow',
#              'Beta',
#              'Held by Insiders',
#              'Held by Institutions',
#              'Shares Short (as of',
#              'Short Ratio',
#              'Short % of Float',
#              'Shares Short (prior ']

def Build_Data_Set():
    data_df = pd.read_csv("electricity-normalized.csv")
    data_df = classify(data_df)
    #data_df = pd.read_csv("key_stats.csv")

    X = np.array(data_df[FEATURES].values)#.tolist())
    #k_fold = KFold(n_splits=10)

    #y = (data_df["Status"].replace("outperform", 1).replace("underperform", 0).values.tolist())
    y = (data_df["class"].replace("UP", 1).replace("DOWN", 0).values.tolist())

    X = preprocessing.scale(X)

    return X,y

def classify(df):
    prev_price = 0
    for i,r in df.iterrows():
        try:
            if r['nswprice'] <= df.at[i+1,'nswprice']:
                df.at[i,'class'] = 'UP'
            else:
                df.at[i,'class'] = 'DOWN'
        except:
            df.at[i,'class'] = 'DOWN'


    #print(df.head())
    return df

def Analysis():


    X, y = Build_Data_Set()
    svc = svm.SVC(C=1, kernel='linear')
    k_fold = KFold(n_splits=10)

    # for train_indices, test_indices in k_fold.split(X):
    #     print ('Train: %s | test: %s' %(train_indices, test_indices))

    crossVal = cross_val_score(svc, X, y, cv=k_fold, n_jobs=-1)
    print(crossVal)
    print('Mean value: %s' %(statistics.mean(crossVal)))


    # test_size = int(floor(len(X)*0.2))
    # print(len(X))
    #
    #
    # clf = svm.SVC(kernel="linear", C = 1.0)
    # clf.fit(X[:-test_size],y[:-test_size])
    #
    # correct_count = 0
    # predictions = clf.predict(X)
    # print(predictions)
    # for x in range(0, test_size):
    #     if (predictions[x+len(X)-test_size] == y[x+len(X)-test_size]):
    #         correct_count += 1
    #
    #
    # print("Accuracy:", (correct_count/test_size) * 100.00)


Analysis()
