#ロジスティック回帰を使ってあやめデータを学習させる
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#データの読み込み
iris = datasets.load_iris()

#ラベル値が2をとるデータを捨てる
data = iris.data[iris.target != 2]
target = iris.target[iris.target != 2]

#ロジスティック回帰による学習と交差検定による評価
logi = LogisticRegression()
scores = cross_val_score(logi, data, target, cv=5)	#データを5つに分割して全てのデータを評価用に使用する

#結果表示
print(scores)