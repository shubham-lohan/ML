import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression
import time


def pre_process():
    np.random.seed(0)
    df = pd.read_csv('dataset/diabetes2.csv', sep=',')
    # print(df.head(10))
    df['Pregnancies'].replace(
        to_replace=0, value=df['Pregnancies'].median(), inplace=True)
    df['Glucose'].replace(
        to_replace=0, value=df['Glucose'].median(), inplace=True)
    df['BloodPressure'].replace(
        to_replace=0, value=df['BloodPressure'].median(), inplace=True)
    df['SkinThickness'].replace(
        to_replace=0, value=df['SkinThickness'].median(), inplace=True)
    df['Insulin'].replace(
        to_replace=0, value=df['Insulin'].median(), inplace=True)
    df['BMI'].replace(
        to_replace=0, value=df['BMI'].median(), inplace=True)
    # print(df.head(10))
    # print(df.dtypes)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X = (X-X.mean(axis=0))/X.std(axis=0)
    X_pd = pd.DataFrame(X)
    # print(X_pd,sep='\n-----------------\n')
    # X = X/np.linalg.norm(X)
    m = X.shape[0]
    m_train = int(m*0.7)
    m_val = int(((m-m_train)*2)/3)
    m_test = m-m_train-m_val
    # print(m_train, m_val, m_test)
    data_set = [X, y]
    train_set = [X[:m_train], y[:m_train]]
    val_set = [X[m_train:-m_test], y[m_train:-m_test]]
    test_set = [X[-m_test:], y[-m_test:]]
    # print(len(train_set[0]), len(val_set[0]), len(test_set[0]))
    # print(test_set[0])
    train_pd = pd.DataFrame(train_set[0])
    # print(train_pd)
    return train_set, val_set, test_set


class Logistic_Regression:
    def __init__(self) -> None:
        """
            Constructor:  intitialize value of learning rate and total no of iteration
        """
        self.learning_rate = 0.01
        self.total_iterations = 1000

    def sigmoid(self, z):
        sigm = 1.0/(1.0+np.exp(-z))
        # print(sigm)
        return sigm

    def accuracy(self, y, y_hat):
        m = len(y)
        # if probability of getting 1 is greater than or equal to 0.5 then predict 1
        y_hat[y_hat >= 0.5] = 1
        # if probability of getting 1 is less than 0.5 then predict 0
        y_hat[y_hat < 0.5] = 0
        # returns the average of difference of values in both arrays
        return (1-sum(abs(y-y_hat))/m)*100

    def cost_function(self, X, y, theta):
        """
        Calculate the cost based on predictions.
        """
        m = y.shape[0]
        activ = self.sigmoid(X.dot(theta))
        activ[activ == 0] = 1e-10
        activ[activ == 1] -= 1e-10
        # # print(X.dot(theta))
        # # Calculates X` * ( sigmoid(X*theta) - y )
        derv = ((X.T).dot(activ-y))

        # # calculates the cross entropy loss
        err = (-y * np.log(activ) - (1 - y) * np.log(1 - activ)).mean()
        return derv, err

    def BGD(self, X, y, X_validate, y_validate):
        """
        Finding theta using the gradient descent method
        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        Returns
        -------
        theta : Calculated value of theta on given test set (X,y) with learning rate alpha
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_validate = np.hstack((np.ones((X_validate.shape[0], 1)), X_validate))
        theta = np.zeros((X.shape[1],))
        train_loss_list = []
        val_loss_list = []
        m = len(y)
        for it in range(self.total_iterations+1):

            train_derv, train_loss = self.cost_function(X, y, theta)
            val_derv, val_loss = self.cost_function(
                X_validate, y_validate, theta)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            theta = theta-self.learning_rate*train_derv

        return theta, train_loss_list, val_loss_list

    def SGD(self, X, y, X_validate, y_validate):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_validate = np.hstack((np.ones((X_validate.shape[0], 1)), X_validate))
        theta = np.zeros((X.shape[1],))
        train_loss_list = []
        val_loss_list = []
        m = y.shape[0]
        for i in range(self.total_iterations+1):
            # selecting one random row for SGD
            train_selection = np.random.randint(0, len(X)-1)
            temp_train_X = []
            temp_train_X.append(X[train_selection])
            temp_train_y = []
            temp_train_y.append(y[train_selection])
            temp_train_X = np.array(temp_train_X)
            temp_train_y = np.array(temp_train_y)
            test_selection = np.random.randint(0, len(X_validate)-1)
            temp_test_X = []
            temp_test_X.append(X_validate[test_selection])
            temp_test_y = []
            temp_test_y.append(y_validate[test_selection])
            temp_test_X = np.array(temp_test_X)
            temp_test_y = np.array(temp_test_y)
            train_derv, train_loss = self.cost_function(
                temp_train_X, temp_train_y, theta)
            val_derv, val_loss = self.cost_function(
                temp_test_X, temp_test_y, theta)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            theta = theta-self.learning_rate*train_derv

        return theta, train_loss_list, val_loss_list

    def plot_loss(self, training_loss, validation_loss, gradient='BGD'):
        plt.plot(training_loss, color="g",
                 label="%s Training Loss at α = %.4g" % (gradient, self.learning_rate))
        plt.plot(validation_loss, color="r",
                 label="%s Validation Loss at α = %.4g" % (gradient, self.learning_rate))
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(
            "plots/Training Loss and Validation Loss vs Iteration for %s at α = %.4g.png" % (gradient, self.learning_rate))
        # plt.show()
        time.sleep(1)

    def comp_ans_diff_alphas(self, train_set, validate_set, alphas=[0.0005, 0.01, 0.0001, 0.005]):
        for alpha in alphas:
            self.learning_rate = alpha
            train_theta_BGD, train_loss_BGD, validate_loss_BGD = regression.BGD(
                train_set[0], train_set[1], validate_set[0], validate_set[1])
            # print(train_theta_BGD)
            train_theta_SGD, train_loss_SGD, validate_loss_SGD = regression.SGD(
                train_set[0], train_set[1], validate_set[0], validate_set[1])
            self.plot_loss(train_loss_BGD, validate_loss_BGD)
            self.plot_loss(train_loss_SGD, validate_loss_SGD, 'SGD')

    def predict(self, X, theta):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_hat = self.sigmoid(X.dot(theta))
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        # return the numpy array y which contains the predicted values
        return y_hat

    def Final_Score(self, y, y_hat):
        m = len(y)
        fp = 0
        fn = 0

        tp = 0
        tn = 0
        for y, y_pred in zip(y, y_hat):
            if y_pred == y:
                if y_pred == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if y_pred == 1:
                    fp += 1
                else:
                    fn += 1
        Confusion_matrix = np.array([[tn, fp], [fn, tp]])

        accuracy = np.divide((tp+tn), m)

        precision = np.divide(tp, (tp+fp))

        recall = np.divide(tp, (tp+fn))

        f1 = 2*(precision * recall)/(precision+recall)

        return Confusion_matrix, accuracy, precision, recall, f1


if __name__ == '__main__':
    train_set, validate_set, test_set = pre_process()
    regression = Logistic_Regression()
    # regression.learning_rate = 0.005
    train_theta_BGD, train_loss_BGD, validate_loss_BGD = regression.BGD(
        train_set[0], train_set[1], validate_set[0], validate_set[1])
    # print(train_theta_BGD)
    train_theta_SGD, train_loss_SGD, validate_loss_SGD = regression.SGD(
        train_set[0], train_set[1], validate_set[0], validate_set[1])
    # regression.plot_loss(train_loss_BGD, validate_loss_BGD)
    # regression.plot_loss(train_loss_SGD, validate_loss_SGD, 'SGD')
    regression.comp_ans_diff_alphas(train_set, validate_set)

    # testing for BGD
    y_hat = regression.predict(test_set[0], train_theta_BGD)
    y = test_set[1]
    Confusion_matrix, accuracy, precision, recall, f1 = regression.Final_Score(
        y, y_hat)
    print("Using BGD")
    print("Confusion Matrix ", Confusion_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1:", f1)

    # testing for SGD
    y_hat = regression.predict(test_set[0], train_theta_SGD)
    y = test_set[1]
    Confusion_matrix, accuracy, precision, recall, f1 = regression.Final_Score(
        y, y_hat)
    print("Using SGD")
    print("Confusion Matrix ", Confusion_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1:", f1)
    # sgd = SGDClassifier(loss="log", alpha=alpha,verbose=1)
    # sgd.fit(train_set[0], train_set[1])
    # # sgd.loss_functions
    # y_hat = sgd.predict(test_set[0])
    # y = test_set[1]
    # Confusion_matrix, accuracy, precision, recall, f1 = regression.Final_Score(
    #     y, y_hat)
    # print(Confusion_matrix, accuracy, precision, recall, f1)
    log = SGDClassifier(loss="log", alpha=0.01, max_iter=1000, verbose=1)
    log.fit(test_set[0], test_set[1])
    print("Number epochs to converge of sklearn's implimentation:", log.n_iter_)
    logistic = LogisticRegression(max_iter=1000)

    # print(logistic.n_iter_)
    logistic.fit(train_set[0], train_set[1])
    y_hat = logistic.predict(test_set[0])
    y = test_set[1]
    Confusion_matrix, accuracy, precision, recall, f1 = regression.Final_Score(
        y, y_hat)
    print("Using Sklearn's Logistic Regression")
    print("Confusion Matrix ", Confusion_matrix)
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("f1:",f1)


