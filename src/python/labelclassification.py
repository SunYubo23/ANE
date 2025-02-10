from __future__ import print_function
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):
    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ['micro', 'macro']
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)

        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)

        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def load_embeddings(filename):
    vectors = np.loadtxt(filename, delimiter=',')
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(int(vec[0]))
        Y.append(vec[1:])
    fin.close()
    return X, Y


if __name__ == '__main__':
    dataset="wiki"
    embfile = 'data/embs/' + dataset + '-u.csv'
    labelfile = 'data/label/' + dataset + '.txt'

    clf_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    vectors1 = load_embeddings(embfile)
    X, Y = read_node_label(labelfile)
    print(embfile)
    z1=np.zeros(5)
    round=10
    for l in range(round):
        cnt=0
        for r in clf_ratio:
            clf = Classifier(vectors=vectors1, clf=LogisticRegression())
            results1 = clf.split_train_evaluate(X, Y, r,seed=int(time.time()))
            # print(results['micro'])
            z1[cnt]+=results1['micro']
            cnt+=1
    print(z1/round)