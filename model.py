import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

d = load_iris()
X = d.data
y = d.target

m = RandomForestClassifier().fit(X, y)
pickle.dump(m, open('model.p', 'wb'))
