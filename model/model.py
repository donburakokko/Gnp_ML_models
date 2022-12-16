from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os

#データも書き換える
iris_df = sns.load_dataset('iris') # データセットの読み込み
iris_df = iris_df[(iris_df['species']=='versicolor') | (iris_df['species']=='virginica')] # 簡単のため、2品種に絞る
x = iris_df.drop('species', axis=1)
y = iris_df['species'].map({'versicolor': 0, 'virginica': 1})

#グリッドサーチ
kf = KFold(n_splits=5, shuffle=True, random_state=1)
def logistic():
    name = "logistic"

    auc_ = []
    accuracy = []
    precision = []
    recall = []
    f1 = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()
    #i = 0

    #条件設定
    max_score = 0
    SearchMethod = 0
    LR_grid = {LogisticRegression(): {"C": [10 ** i for i in range(-5, 6)], "random_state": [i for i in range(0, 101)]}}
    #LR_random = {LogisticRegression(): {"C": scipy.stats.uniform(0.00001, 1000),"random_state": scipy.stats.randint(0, 100)}}

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        train_x = x.iloc[train_index]
        test_x  = x.iloc[test_index]
        train_y = y.iloc[train_index]
        test_y  = y.iloc[test_index]

        #このモデル部分を変更する
        for model, param in LR_grid.items():
        clf = GridSearchCV(model, param)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        score = f1_score(test_y, pred_y, average="micro")

        if max_score < score:
            max_score = score
            best_param = clf.best_params_
            best_model = model.__class__.__name__
        print(best_param)

        clf = LogisticRegression(C = best_param['C'], random_state = best_param['random_state'])
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        proba_y = clf.predict_proba(test_x)[: , 1]


        viz = plot_roc_curve(clf, test_x, test_y, 
                            name='ROC fold {}'.format(i+1), 
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        #フォールドごとの各スコア算出
        print(name+'・fold'+str(i+1)+'_start')
        score = roc_auc_score(test_y, pred_y)
        auc_.append(score)
        print('AUC' ,score)
        acc = accuracy_score(test_y, pred_y)
        accuracy.append(acc)
        print('accuracy:', acc)
        pre = precision_score(test_y, pred_y)
        precision.append(pre)
        print('precision:', pre)
        rec = recall_score(test_y, pred_y)
        recall.append(rec)
        print('recall:', rec)
        f = f1_score(test_y, pred_y)
        f1.append(f)
        print('f1_score:', f)
        print('finish\n')

    #このままだとcwdがpathになってるので、statisticsに書き換えること
    now = datetime.datetime.now()
    path = os.getcwd()
    na =  name + "_" + now.strftime('%Y%m%d_%H%M%S')
    filename = path + "/" + na + '.txt'

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    #ここからコピペ
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="ROC curve")
    ax.legend(loc="lower right")
    #このままだとcwdがpathになってるので、statisticsに書き換えること
    plt.savefig(na +"_ROC.png", format="png")
    plt.show()


    print('以下、平均スコア')
    print('平均AUC',np.mean(auc_))
    print('平均accuracy',np.mean(accuracy))
    print('平均precision',np.mean(precision))
    print('平均recall',np.mean(recall))
    print('平均F1',np.mean(f1))

    print(filename)
    with open(filename, mode='w') as f:
        f.write('以下、平均スコア\n')
        f.write('平均AUC '+str(np.mean(auc_))+"\n")
        f.write('平均accuracy '+str(np.mean(accuracy))+"\n")
        f.write('平均precision '+str(np.mean(precision))+"\n")
        f.write('平均recall '+str(np.mean(recall))+"\n")
        f.write('平均F1 '+str(np.mean(f1))+"\n")
        f.close()

if __name__ == '__main__':
  logistic()