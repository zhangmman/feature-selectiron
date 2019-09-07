'''
数字分类任务中像素相关性的递归特征消除示例。
Recursive feature elimination
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


# Load the digits dataset
digits = load_digits()
X1 = digits.images.reshape((len(digits.images), -1))
y1 = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe1 = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe1.fit(X1, y1)
ranking1 = rfe1.ranking_.reshape(digits.images[0].shape)


iris = load_iris()
# Add the noisy data to the informative features
X2 = iris.data
y2 = iris.target
X2_indices = np.arange(X2.shape[-1])
rfe2 = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe2.fit(X2, y2)
support = rfe2.support_
ranking2 = rfe2.ranking_  # 返回的是一个特征变量的重要性排序  4,3,2,1


# Plot pixel ranking
plt.matshow(ranking1, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")


plt.figure()
plt.subplot(1, 2, 1)
plt.bar(X2_indices, 1/ranking2, width=.2, label='重要性排序', color='navy')
plt.title("# ERF: 1/ranking2 ")















plt.show()
