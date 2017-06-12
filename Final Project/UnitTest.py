import numpy as np
from MultinomialNB import MultinomialNB

X = np.array([
    [2, 1, 0, 0, 0, 0],
    [2, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 1]
])
y = np.array([0, 0, 0, 1])
nb = MultinomialNB().fit(X, y)

X_test = np.array([[3, 0, 0, 0, 1, 1], [0, 1, 1, 0, 1, 1]])
print(nb.predict(X_test))
