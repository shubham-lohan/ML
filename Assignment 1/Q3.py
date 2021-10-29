import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, roc_curve, precision_score

"""
references: 
https://scikit-learn.org/stable/modules/naive_bayes.html
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/#:~:text=A%20useful%20tool%20when%20predicting,values%20between%200.0%20and%201.0.
"""


def pre_process():
    np.random.seed(0)
    data_train = pd.read_csv(
        'dataset/fashion-mnist_train.csv', sep=',').to_numpy()
    # print(df_train.head)
    data_test = pd.read_csv(
        'dataset/fashion-mnist_test.csv', sep=',').to_numpy()
    data_train = data_train[np.logical_or(
        data_train[:, 0] == 1, data_train[:, 0] == 2)]
    data_test = data_test[np.logical_or(
        data_test[:, 0] == 1, data_test[:, 0] == 2)]
    X_train, y_train = data_train[:, 1:], data_train[:, 0]
    X_test, y_test = data_test[:, 1:], data_test[:, 0]
    binarizer = Binarizer(threshold=127)

    X_train = binarizer.fit_transform(X_train)
    # print(X_train)
    # y_train = binarizer.transform(y_train)
    X_test = binarizer.fit_transform(X_test)
    # y_test = binarizer.transform(y_test)
    return X_train, y_train, X_test, y_test


class Naive_Bayes:

    def __init__(self) -> None:
        self.mean = []
        self.variance = []
        self.classes = []
        self.priors = []

    def scratch(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, n_features))
        self.variance = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            # mean for class c
            self.mean[i, :] = X_c.mean(axis=0)
            # varaince for class c
            self.variance[i, :] = X_c.var(axis=0)
            # variance for class c
            self.priors[i] = X_c.shape[0] / float(n_samples)
        # return classes,mean,variance,priors
        self.mean[self.mean == 0] = 1e-10
        self.variance[self.variance == 0] = 1e-10
        # print(self.classes, self.mean, self.variance, self.priors)

    def classify_sample(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            m = self.mean[i]
            v = self.variance[i]
            # print(v)
            numerator = np.exp(-(x-m)**2 / (2 * v))
            denominator = np.sqrt(2 * np.pi * v)
            frac = numerator / denominator
            frac[frac == 0] = 1e-10
            post = np.sum(np.log(frac))
            post += prior
            posteriors.append(post)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.classify_sample(x))
        return np.array(y_pred)

    def accuracy(self, y, y_hat):
        count = np.count_nonzero(y == y_hat)
        return count/len(y)

    def cross_validation(self, X, y, k=10):
        subset_size = len(X)//k
        accuracy = []
        for i in range(k):
            X_testing_this_round = X[i*subset_size:]
            y_testing_this_round = y[i*subset_size:]
            l = list(X[(i+1)*subset_size:])
            l.append(X[:i * subset_size])
            X_training_this_round = np.array(l, dtype=np.float64)
            l = list(y[(i+1)*subset_size:])
            l.append(y[:i * subset_size])
            y_training_this_round = np.array(l, dtype=np.float64)
            self.scratch(X_training_this_round, y_training_this_round)
            y_hat = self.predict(X_testing_this_round, y_testing_this_round)
            accur = self.accuracy(y_test, y_hat)
            accuracy.append(accur)
        return np.array(accuracy).mean()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = pre_process()
    # print(X_train.shape)
    bayes = Naive_Bayes()
    bayes.scratch(X_train, y_train)
    y_hat = bayes.predict(X_test)
    accur = bayes.accuracy(y_test, y_hat)
    print("Using Scratch code:")
    print("Accuracy:", accur)
    # print(y_hat)
    # Sklearn's Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_hat = gnb.predict(X_test)
    accur = accuracy_score(y_test, y_hat)
    print("Sklearn's Implimentation:")
    print(accur)

    precision = precision_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    print("Precision", precision)
    print("Recall", recall)
    conf_matrix = confusion_matrix(y_test, y_hat)
    print("Confusion Matrix: ", conf_matrix)

    # ROC curve
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = gnb.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # summarize scores
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs, pos_label=2)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs, pos_label=2)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes')
    # axis labels
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.savefig("plots/ROC Curve.png")
    plt.legend()
    # show the plot
    # plt.show()

    # accur = bayes.cross_validation(X_test, y_test)
    # print(accur)

