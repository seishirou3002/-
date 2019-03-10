#人工的データを使った線形回帰プログラム
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

#線形回帰に使用するテストデータを乱数によって生成
np.random.seed(0)												#何度実行しても生成される乱数をを同じにしている
regdata = datasets.make_regression(100, 1, noise= 20.0)			#ノイズの大きさを20に設定した1次元のデータを100のサンプル数で生成

#学習を行いパラメータを表示
lin = linear_model.LinearRegression()							#線形回帰クラスのインスタンス生成
lin.fit(regdata[0],regdata[1])									#フィッティング計算
print("coef and intercept :",lin.coef_, lin.intercept_)
print("score :", lin.score(regdata[0], regdata[1]))

#グラフを描画
xr = [-2.5,2.5]
plt.plot(xr, lin.coef_ * xr + lin.intercept_)					#パラメータ値を直線(y = a * x + b)に当てはめる
plt.scatter(regdata[0],regdata[1])								#点群をプロット

plt.show()