#SVMを使ってあやめデータを学習させる
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

#あやめデータの読み込み
iris = datasets.load_iris()

#学習
svc = svm.SVC()
scores = cross_val_score(svc,iris.data,iris.target,cv=5)

#結果表示
print(scores)
print("Accuracy:", scores.mean())