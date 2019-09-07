from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

iris = load_iris()
X, y = iris.data, iris.target
print('X.shape is:{}'.format(X.shape))
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new = SelectKBest(chi2, k=2).fit(X, y)   # 返回选择表里的一些属性，p值等
X_new = SelectKBest(chi2, k=2).fit(X, y).get_support(indices=True)  # 返回选择变量的序号

print('X_new.shape is:{}'.format(X_new.shape))


