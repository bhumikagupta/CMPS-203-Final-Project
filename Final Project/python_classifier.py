import csv
from pprint import pprint
import numpy as np
import nltk
from nltk.corpus import stopwords
# from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import collections
import  operator


def preprocess(file):
    # reading as csv
    with open(file, 'r') as tsv:
        spam_new = [line.strip().split('\t') for line in tsv]

    # Get the number of reviews based on the dataframe column size
    num_sms = len(spam_new)

    clean_data = []
    labels = []

    for i in range(0, num_sms):
        if spam_new[i][0] == 'spam':
            labels.append(1)
        else:
            labels.append(0)
        clean_data.append((clean(spam_new[i][1])))

    return clean_data, labels


def clean(text):
    # removing html
    # without_html = BeautifulSoup(text).get_text()

    # removing special characters
    not_so_special = re.sub("[^a-zA-Z]", " ", text)

    # to lowercase
    lower_case_words = not_so_special.lower().split()

    # removing stop words

    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in lower_case_words if w not in stops]

    return " ".join(lower_case_words)


def featurize(data, labels):
    ratio = 0.9
    split_point = int(ratio * len(data))

    X = data[:split_point]
    y = labels[:split_point]

    x_test = data[split_point:]
    y_test = labels[split_point:]

    no_of_features = 1000
    features = vocab(X, no_of_features)
    v = [pair[0] for pair in features]

    train_data_features = make_vector(X,v)
    print(y)
    test_data_features = make_vector(x_test,v)

    return train_data_features, test_data_features, y, y_test


def classify(train_features, test_features, train_labels, test_labels):

    nb = MultinomialNB()
    nb = nb.fit(train_features, train_labels)
    y_pred = nb.predict(test_features)

    print(confusion_matrix(test_labels, y_pred))
    print(accuracy_score(test_labels, y_pred))

def vocab(data, no_of_features):
    text = ' '.join(data)
    c = collections.Counter(text.split())
    return c.most_common()


def shuffle(data, labels):
    joint_list = list(zip(data, labels))
    np.random.shuffle(joint_list)
    data, labels = zip(*joint_list)
    return data, labels

def make_vector(data, vocab):
    bag_of_words = []
    for message in data:
        word_freqs = [0]*len(vocab)
        tokens = message.split()
        for i in range(len(tokens)):
            if tokens[i] in vocab:
                word_freqs[vocab.index(tokens[i])] += 1
        # print(word_freqs)
        bag_of_words.append(word_freqs)
    # print(bag_of_words)
    return np.array(bag_of_words)

def main():
    clean_file, labels = preprocess("Spam")
    # clean_file, labels = shuffle(clean_file, labels)
    train_features, test_features, train_labels, test_labels = featurize(clean_file, labels)
    classify(train_features, test_features, train_labels, test_labels)


if __name__ == "__main__":
    main()
