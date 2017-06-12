import numpy as np
from pprint import pprint


class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # group by class
        classes = np.unique(y)
        separated = []
        for category in classes:
            separated.append([x for x, t in zip(X, y) if t == category])
        # print(separated)

        # the prior log probability for each class.
        count_sample = X.shape[0]
        self.class_log_prior_ = []

        for i in separated:
            self.class_log_prior_.append(np.log(len(i) / count_sample))

        # count each word for each class and add self.alpha as smoothing
        count = []
        for i in separated:
            count.append(np.array(i).sum(axis=0))
        count = np.array(count) + self.alpha
        print('count each word for each class with smoothing\n', count)

        # calculate the log probability of each word
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        print('calculate the log probability of each word\n', self.feature_log_prob_)

        return self

    def predict(self, X: object) -> object:
        prediction = []
        for x in X:
            prediction.append((self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_)
        # print(prediction)
        return np.argmax(prediction, axis=1)
