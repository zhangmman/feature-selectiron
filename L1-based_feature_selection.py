'''
通过 L1-based feature selection 选择变量
'''
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel


# 案例一
iris = load_iris()
X, y = iris.data, iris.target

print('X.shape {}'.format(X.shape))

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print('X_new.shape {}'.format(X_new.shape))
