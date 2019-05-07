import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import pandas as pd
from matplotlib import style
from math import floor
import statistics
style.use("ggplot")

FEATURES = ["date","period","nswprice","nswdemand"]

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

    return df

def Analysis():

    folds = 10
    X, y = Build_Data_Set()
    test_size = int(floor(len(X)*0.2))
    y_eval_test = y[len(X)-test_size:]
    X_eval_test = X[len(X)-test_size:]
    print('\n Building cross validation')
    svc = svm.SVC(C=1, kernel='rbf', gamma=0.1, probability=True)
    k_fold = KFold(n_splits=folds)


    # print('\n Cross validation indices')
    # for train_indices, test_indices in k_fold.split(X[:-test_size]):
    #     print ('Train: %s | test: %s' %(train_indices, test_indices))




    X_folds = np.array_split(X[:-test_size], 10)
    y_folds = np.array_split(y[:-test_size], 10)
    scores = list()
    final_eval = list()
    aucs = list()
    # for k in range(10):
    #     # We use 'list' to copy, in order to 'pop' later on
    #     X_train = list(X_folds)
    #     X_test = X_train.pop(k)
    #     X_train = np.concatenate(X_train)
    #     y_train = list(y_folds)
    #     y_test = y_train.pop(k)
    #     y_train = np.concatenate(y_train)
    #     scores.append(svc.fit(X_train, y_train, ).score(X_test, y_test))
    #     predictions = svc.predict(X_eval_test)
    #     correct_count = 0
    #     for x in range(0,test_size):
    #         if (predictions[x] == y[x+len(X)-test_size]):
    #             correct_count += 1
    #     acc = correct_count/test_size
    #     final_eval.append(acc)
    #         # Confusion matrix
    #     con_matrix = metrics.confusion_matrix(y_eval_test,predictions)
    #     print('Fold: %s \n'%(k+1),con_matrix)
    #     FPR, TPR, thresholds = metrics.roc_curve(y_eval_test, predictions)
    #     print(thresholds)
    #     roc_auc = metrics.auc(FPR, TPR)
    #     aucs.append(roc_auc)
    #     plt.figure(k+1)
    #     print(FPR)
    #     plt.plot(FPR, TPR, color='b', label='ROC fold %d (AUC = %0.2f)' % (k+1, roc_auc))
    #     plt.plot([0,1], [0,1], linestyle='--', color='r')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.0])
    #     plt.title('ROC curve for best model, fold %s' %(k+1))
    #     plt.xlabel('False Positive rate')
    #     plt.ylabel('True positive rate')
    #     plt.legend(loc='lower right')
    #     plt.grid(True)
    # plt.show()
    #
    # for i in range(10):
    #     print('\nFold %s Validation Accuracy: %s . Evalutation Accuracy: %s . AUC: %s' %(i+1, scores[i], final_eval[i], aucs[i]))

    # ROC curve of best model
    k = 8
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    svc.fit(X_train, y_train, )
    pred = svc.predict(X_eval_test)

    con_matrix = metrics.confusion_matrix(y_eval_test,pred)
    TP = con_matrix[1,1]
    TN = con_matrix[0,0]
    FP = con_matrix[0,1]
    FN = con_matrix[1,0]
    c_acc = (TP + TN) / float(TP + TN + FP + FN)
    c_err = FP + FN / float(TP + TN + FP + FN)
    sensitivity = TP / float(TP+FN)
    specificity = TN / float(TN+FP)
    FP_rate = FP / float(TN + FP)

    print('Fold: %s \nConfusion Matrix\n%s\nSensitivity: %s, Specificity: %s\n'%(k+1,con_matrix, sensitivity, specificity))

    FPR, TPR, thresholds = metrics.roc_curve(y_eval_test, pred)
    roc_auc = metrics.auc(FPR, TPR)
    print(roc_auc)
    plt.plot(FPR, TPR, color='b', label='ROC fold %d (AUC = %0.2f)' % (k+1, roc_auc))
    plt.plot([0,1], [0,1], linestyle='--', color='r')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt .title('ROC curve for best model, fold %s' %(k+1))
    plt.xlabel('False Positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()



    # print('\n Training models')
    # crossVal = cross_val_score(svc, X[:-test_size], y[:-test_size], cv=k_fold, n_jobs=-1)
    # highVal = max(crossVal)
    # #highValindex = crossVal.index(max(crossVal))
    # print('\n Scores from cross validation\n',crossVal)
    # print('\n Mean value: %s' %(statistics.mean(crossVal)))
    # print('\n Highest validation: ', highVal)
    # for train, test in k_fold.split(X[:-test_size]):
    #     svc.fit(, y[train])


    # print('Final evaluation')
    # correct_count = 0
    # predictions = svc.predict(X)
    # print(predictions)
    # for x in range(0, test_size):
    #     if (predictions[x+len(X)-test_size] == y[x+len(X)-test_size]):
    #         correct_count += 1
    #
    #
    # print("Accuracy:", (correct_count/test_size) * 100.00)


Analysis()
