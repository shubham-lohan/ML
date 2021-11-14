import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier


def pre_process():
    # pass
    df = pd.read_csv('dataset/PRSA_data_2010.1.1-2014.12.31.csv')
    # print(df)
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop('No', axis=1, inplace=True)
    df['pm2.5'].replace(to_replace=np.NaN,
                        value=df['pm2.5'].median(), inplace=True)
    # encoding data into integer values
    df['cbwd'] = df['cbwd'].replace('NW', 1)
    df['cbwd'] = df['cbwd'].replace('cv', 2)
    df['cbwd'] = df['cbwd'].replace('SE', 3)
    df['cbwd'] = df['cbwd'].replace('NE', 4)

    y = df['month'].values
    df.drop('month', axis=1, inplace=True)
    X = df.values
    n = X.shape[0]
    X_train = X[:int(n*0.7)]
    y_train = y[:int(n*0.7)]

    # print(df)
    X_value = X[int(n*0.7):-int(n*0.15)]
    y_value = y[int(n*0.7):-int(n*0.15)]

    X_test = X[-int(n*0.15):]
    y_test = y[-int(n*0.15):]

    train_data = [X_train, y_train]
    validate_data = [X_value, y_value]
    test_data = [X_test, y_test]
    return train_data, validate_data, test_data


def accuracy_score(y_true, y_predicted):
    return (y_true == y_predicted).mean()


class DecisionTree:
    def do_something(self, train_data, validate_data, test_data, impurity_type='gini'):
        X_train, y_train = train_data
        X_test, y_test = test_data
        clf = DecisionTreeClassifier(criterion=impurity_type)
        clf.fit(X_train, y_train)
        giniprediction = clf.predict(X_test)
        Acc = accuracy_score(y_test, giniprediction)
        print("Testing Accuracy using", impurity_type, Acc)
        print("Training Accuracy using", impurity_type,
              clf.score(X_train, y_train))

    def bestheight(self, train_data, validate_data, test_data):
        X_train, y_train = train_data
        X_test, y_test = test_data
        depth = [2, 4, 8, 10, 15, 30]
        A = {}
        A_train = {}
        for i in depth:
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
            clf.fit(X_train, y_train)
            giniprediction = clf.predict(X_test)
            Acc = accuracy_score(y_test, giniprediction)

            train_giniprediction = clf.predict(X_train)
            train_Acc = accuracy_score(y_train, train_giniprediction)

            A[i] = Acc
            A_train[i] = train_Acc
        key_list = list(A.keys())
        val_list = list(A.values())
        position = val_list.index(max(val_list))
        # print(key_list[position])
        # print(max(val_list))
        print(f"Best Testing Accuracy is %.10f with the depth %d" %
              (max(val_list), key_list[position]))
        # print(k[v.index(max(v))])
        plt.plot(depth, A.values(), label='Test', marker='o')
        plt.plot(depth, A_train.values(), label='Train', marker='o')
        plt.xlabel('depth')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("plots/Q1.png")
        plt.show()

    def ensembling(self, train_data, validate_data, test_data):
        num_stumps = 100
        n, m = train_data[0].shape
        new_train_data = np.hstack(
            (train_data[0], train_data[1].reshape(n, -1)))

        majority_votes = np.zeros((test_data[1].shape[0], 13), dtype=int)

        for i in range(num_stumps):
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            np.random.shuffle(new_train_data)
            temp = new_train_data[:int(n*0.5)]
            X, y = temp[:, :-1], temp[:, -1]
            tree = tree.fit(X, y)
            y_hat = tree.predict(test_data[0])
            for j in range(test_data[0].shape[0]):
                majority_votes[j][int(y_hat[j])] += 1
        final_yhat = np.argmax(majority_votes, axis=1)
        # print(final_yhat.shape,test_data[1].shape)
        print("Accuracy", accuracy_score(test_data[1], final_yhat))

    def dancing_ensembling(self, train_data, validate_data, test_data):
        num_stumps = [20, 50, 100]
        depths = [4, 8, 10, 15, 20, 30]
        n, m = train_data[0].shape
        new_train_data = np.hstack(
            (train_data[0], train_data[1].reshape(n, -1)))
        for depth in depths:
            for stump in num_stumps:
                majority_votes_test = np.zeros(
                    (test_data[1].shape[0], 13), dtype=int)
                majority_votes_train = np.zeros(
                    (train_data[1].shape[0], 13), dtype=int)
                majority_votes_val = np.zeros(
                    (validate_data[1].shape[0], 13), dtype=int)
                for i in range(stump):
                    tree = DecisionTreeClassifier(
                        criterion='entropy', max_depth=depth)
                    np.random.shuffle(new_train_data)
                    temp = new_train_data[:int(n*0.5)]
                    X, y = temp[:, :-1], temp[:, -1]
                    tree = tree.fit(X, y)
                    y_hat_test = tree.predict(test_data[0])
                    y_hat_train = tree.predict(train_data[0])
                    y_hat_val = tree.predict(validate_data[0])
                    for j in range(test_data[0].shape[0]):
                        majority_votes_test[j][int(y_hat_test[j])] += 1
                    for j in range(train_data[0].shape[0]):
                        majority_votes_train[j][int(y_hat_train[j])] += 1
                    for j in range(validate_data[0].shape[0]):
                        majority_votes_val[j][int(y_hat_val[j])] += 1
                final_yhat_test = np.argmax(majority_votes_test, axis=1)
                final_yhat_train = np.argmax(majority_votes_train, axis=1)
                final_yhat_val = np.argmax(majority_votes_val, axis=1)

                # print(final_yhat_test.shape,test_data[1].shape)
                print(f"Accuracy at testing: %.3f at depth: %d Stumps: %d " %
                      (accuracy_score(test_data[1], final_yhat_test), depth, stump))
                print(f"Accuracy at training: %.3f at depth: %d Stumps: %d " %
                      (accuracy_score(train_data[1], final_yhat_train), depth, stump))
                print(f"Accuracy at validation: %.3f at depth: %d Stumps: %d " %
                      (accuracy_score(validate_data[1], final_yhat_val), depth, stump))

    def boosting(self, train_data, validate_data, test_data):
        esitamators = [4, 8, 10, 15, 20]
        for est in esitamators:
            clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='gini'), n_estimators=est).fit(
                train_data[0], train_data[1])
            y_hat = clf.predict(test_data[0])
            y = test_data[1]
            print(f"Accuracy: %.3f at Estimator: %d " %
                  (accuracy_score(y, y_hat), est))


if __name__ == '__main__':
    train_data, validate_data, test_data = pre_process()
    dtree = DecisionTree()
    dtree.do_something(train_data, validate_data, test_data, 'entropy')
    dtree.bestheight(train_data, validate_data, test_data)
    dtree.ensembling(train_data, validate_data, test_data)
    dtree.dancing_ensembling(train_data, validate_data, test_data)
    dtree.boosting(train_data, validate_data, test_data)
