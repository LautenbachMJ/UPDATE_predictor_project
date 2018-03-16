from sklearn import svm
import numpy as np


loaded = np.load('../numpy/svm_input_10.npz')
aa_array = loaded['aa_onehot_encoded']
top_array = loaded['TOP_num']
#top_array.tolist()

print(aa_array)
print(top_array.tolist())

#print(top_array)
x = aa_array
y = top_array.tolist()
#print(x.shape, y.shape)
#
clf = svm.SVC()
clf.fit(x, y)  
print(clf)


#clf.predict([[2., 2.]])
#array([1])

#http://scikit-learn.org/stable/modules/svm.html
