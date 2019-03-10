#糖尿病のデータを線形回帰で予測したプログラム
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#糖尿病のデータを読み込み
#データの形式：入力(年齢、性別、体重、血圧等)diabetes.data、出力(1年の病気の進行具合)diabetes.target
diabetes = datasets.load_diabetes()

#読み込んだデータを訓練用と評価用に分ける
#取得したデータの最後の20件を評価用にそれ以外を訓練用に分けている
data_train = diabetes.data[:-20]
target_train = diabetes.target[:-20]
data_test = diabetes.data[-20:]
target_test = diabetes.target[-20:]

#学習させる
lin = linear_model.LinearRegression()
lin.fit(data_train, target_train)

#当てはまり度合いを表示
print("Score :", lin.score(data_test, target_test))

#最初の評価用データについて結果を予想して、実際の値と並べて表示
print("Prediction :", lin.predict(data_test)) 	#予想
print("Actual value :", target_test[0])	#実際の値
