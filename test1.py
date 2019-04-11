import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.1, C=100)

x = digits.data[:-10]
y = digits.target[:-10]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[-4].reshape(1,-1)))

plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
