# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# +
#データ読み込み関数
def read_data(data_path):
    data_name = pd.read_csv(filepath_or_buffer=data_path, encoding="utf_8", sep=",")
    return data_name

#判定結果がどちらだったのか判断して記載
def decide_predict(df):
    non_pub = df['not publish']
    pub = df['publish']
    #リストへの変換
    l_non_pub = non_pub.values.tolist()
    l_pub = pub.values.tolist()
    #比較
    predict = []
    for i in range(len(l_non_pub)):
        if l_non_pub[i] > l_pub[i]:
            predict.append(0)
        else:
            predict.append(1)
    return predict

#各モデルによる予測結果とLSTM_Aの結果と掲載したかどうかを比較して数をカウント
def count_data(predictdata):
    #両方正解
    correct = []
    #両方間違い
    incorrect = []
    #前回正解で今回間違い
    n_incorrect = []
    #前回間違いで今回正解
    n_correcct = []

    for i in range(len(l_result)):
        if l_result[i] == L_result[i] and l_result[i] == predictdata[i]:
            correct.append(i)
        elif l_result[i] == L_result[i] and l_result[i] != predictdata[i]:
            n_incorrect.append(i)
        elif l_result[i] != L_result[i] and l_result[i] == predictdata[i]:
            n_correcct.append(i)
        else:
            incorrect.append(i)
   # p_correct = correct/len(l_result)*100
    #p_incorrect = incorrect/len(l_result)*100
    #p_n_incorrect = n_incorrect/len(l_result)*100
    #p_n_correcct = n_correcct/len(l_result)*100
    return [correct, incorrect, n_incorrect, n_correcct]


# -

LSTM_result = read_data("Datas/model1_dA_predict.csv")

Adata = read_data("model2/result/A/result.csv")
Bdata = read_data("model2/result/B/result.csv")
Cdata = read_data("model2/result/C/result.csv")
Ddata = read_data("model3/result/D/result.csv")
Edata = read_data("model4/result/E/result.csv")
Fdata = read_data("model4/result/F/result.csv")
Gdata = read_data("model5/result/G/result.csv")
Hdata = read_data("model4/result/H/result.csv")
Idata = read_data("model5/result/I/result.csv")
Jdata = read_data("model5/result/J/result.csv")
Kdata = read_data("model6/result/K/result.csv")
Ldata = read_data("model7/result/L/result.csv")
Mdata = read_data("model7/result/M/result.csv")
Ndata = read_data("model7/result/N/result.csv")
Odata = read_data("model8/result/O/result.csv")

#実際の結果を抽出
result = Adata['retweet']
l_result = result.values.tolist()
#LSTMのみのモデルの判定結果を抽出
L_result = []
L_result = decide_predict(LSTM_result)

#予測結果の判定
A_predict = decide_predict(Adata)
B_predict = decide_predict(Bdata)
C_predict = decide_predict(Cdata)
D_predict = decide_predict(Ddata)
E_predict = decide_predict(Edata)
F_predict = decide_predict(Fdata)
G_predict = decide_predict(Gdata)
H_predict = decide_predict(Hdata)
I_predict = decide_predict(Idata)
J_predict = decide_predict(Jdata)
K_predict = decide_predict(Kdata)
L_predict = decide_predict(Ldata)
M_predict = decide_predict(Mdata)
N_predict = decide_predict(Ndata)
O_predict = decide_predict(Odata)

Are = count_data(A_predict)
Bre = count_data(B_predict)
Cre = count_data(C_predict)
Dre = count_data(D_predict)
Ere = count_data(E_predict)
Fre = count_data(F_predict)
Gre = count_data(G_predict)
Hre = count_data(H_predict)
Ire = count_data(I_predict)
Jre = count_data(J_predict)
Kre = count_data(K_predict)
Lre = count_data(L_predict)
Mre = count_data(M_predict)
Nre = count_data(N_predict)
Ore = count_data(O_predict)

Are

are = count_data(A_predict)
bre = count_data(B_predict)
cre = count_data(C_predict)
dre = count_data(D_predict)
ere = count_data(E_predict)
fre = count_data(F_predict)
gre = count_data(G_predict)
hre = count_data(H_predict)
ire = count_data(I_predict)
jre = count_data(J_predict)
kre = count_data(K_predict)
lre = count_data(L_predict)
mre = count_data(M_predict)
nre = count_data(N_predict)
ore = count_data(O_predict)

#前回間違いで今回正解
ore[3]

#210 368 717
Adata[1425:]

Bre

Cre

Dre

Ere

Fre

Gre

Hre

Ire

Jre

Kre

Lre

Mre

Nre

Ore
