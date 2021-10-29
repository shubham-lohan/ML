import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV


def pre_process():
    """
    This function is used to preProcessed the data
    Parameters
    ----------
    None

    Returns
    -------
    X_train, y_train, X_test, y_test :numpy array of trainning and testing

    """
    np.random.seed(0)
    df = pd.read_csv('dataset/abalone.data', sep=',', header=None)
    df.sample(frac=1)
    df[0].replace('M', 1, inplace=True)
    df[0].replace('F', 2, inplace=True)
    df[0].replace('I', 3, inplace=True)
    # print(df.dtypes)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # X = (X-X.mean(axis=0))/X.std(axis=0)  # no need to normalize
    m = X.shape[0]
    m_test = int(m*0.8)
    X_train, X_test = X[:m_test], X[m_test:]
    y_train, y_test = y[:m_test], y[m_test:]
    return X, y, X_train, y_train, X_test, y_test


class Linear_Regression:
    def __init__(self):
        """
            Constructor:  intitialize value of learning rate and total no of iteration
        """
        self.learning_rate = 0.01
        self.total_iterations = 1000

    def cost_function(self, X, y, theta):
        """
        find cost value at given parameters
        Parameters:
        x: features
        y: target values
        theta: model parameter
        Returns:
        -------
        cost: cost with current theta
        """
        m = len(y)
        cost = np.sum((((X.dot(theta)) - y) ** 2) / (2*m))
        return cost

    def RMSE(self, y, y_hat):
        """
        This function is used to find the Root Mean Squared Error
        Parameters
        ----------
        y,y_hat : 1-dimensional numpy array of shape (n_samples,)
        Returns
        -------
        error
        """
        diff = y-y_hat
        m = len(y)
        error = np.power((1/m)*np.sum(np.power(diff, 2)), 0.5)
        return error

    def gradient_descent(self, X, y):
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
        theta = np.zeros((X.shape[1],))
        costs = []
        m = len(y)
        for it in range(self.total_iterations+1):
            # Calculate the value -- Forward Propagation
            z = X.dot(theta)
            loss = z - y
            # Calculate gradient descent
            weight_gradient = X.T.dot(loss) / m
            # Update theta
            theta = theta - self.learning_rate*weight_gradient
            cost = self.cost_function(X, y, theta)
            costs.append(cost)
        return theta, costs

    def predict(self, X, theta):
        """
        Predict the value of trained linear model.
        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
        theta : 1-dimensional numpy array of shape (n_samples,)
        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = np.dot(X, theta)
        return y

    def Ridge_lasso_Regression(self, X_train, y_train, X_test, y_test):
        """
        Calculate and Plot the value of RMSE of both ridge and lasso Regression at different values of learning rate alpha
        Parameters
        ----------
        X_train, y_train : training dataset for the model
        X_test, y_test :  testing dataset for the model
        Returns
        -------
        None ;)
        """
        alphas = [(1e-5), 0.0001, 0.001, 0.003, 0.01,
                  0.03, 0.1, 0.25, 0.5, 1, 3, 10]
        rmse_ridge = []
        rmse_lasso = []
        coeff_ridge = []
        coeff_lasso = []
        for alpha in alphas:
            rr = Ridge(alpha=alpha)
            rr.fit(X_train, y_train)
            y_hat_ridge = rr.predict(X_test)
            ll = Lasso(alpha=alpha)
            ll.fit(X_train, y_train)
            y_hat_lasso = ll.predict(X_test)
            rmse_ridge.append(self.RMSE(y_test, y_hat_ridge))
            rmse_lasso.append(self.RMSE(y_test, y_hat_lasso))
            coeff_ridge.append(np.hstack((rr.intercept_, rr.coef_)))
            coeff_lasso.append(np.hstack((ll.intercept_, ll.coef_)))
        best_coeff_ridge = coeff_ridge[rmse_ridge.index(min(rmse_ridge))]
        best_alpha_ridge = alphas[rmse_ridge.index(min(rmse_ridge))]
        best_coeff_lasso = coeff_lasso[rmse_lasso.index(min(rmse_lasso))]
        best_alpha_lasso = alphas[rmse_lasso.index(min(rmse_lasso))]
        print("Ridge best alpha :", best_alpha_ridge)
        print("Ridge best coef :", best_coeff_ridge)
        print("Lasso best alpha :", best_alpha_lasso)
        print("Lasso best coef :", best_coeff_lasso)
        plt.xlabel('alpha')
        plt.ylabel('RMSE')
        plt.plot(alphas, rmse_ridge, marker='o',
                 markersize=7, color='green', label='Ridge Regression')
        plt.plot(alphas, rmse_lasso, marker='o',
                 markersize=7, color='red', label='Lasso Regression')
        plt.xscale('log')
        plt.legend()
        plt.title("Alpha vs RMSE for both Ridge and Lasso Regression")
        plt.savefig("plots/Ridge_lasso_Regression.png")
        plt.show()

    def plot_cost(self, cost):
        """
        Plot the graph of Cost vs Iteration
        """
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.plot(cost)
        plt.title("Cost Vs Iterations")
        plt.savefig("plots/cost vs iteration.png")
        plt.show()

    def linearRegression(self, X_train, y_train, X_test, y_test):
        theta, cost = self.gradient_descent(X_train, y_train)
        print(len(theta), X.shape)
        y_hat_train = self.predict(X_train, theta)
        y_hat_test = self.predict(X_test, theta)
        rmse_train = self.RMSE(y_train, y_hat_train)
        rmse_test = self.RMSE(y_test, y_hat_test)
        return rmse_train, rmse_test, cost

    def Grid_search_function(self, X, y):
        """
        Sklearn’s Grid search function to find the best alpha value and
        the best model coefficient for both Ridge and Lasso Regression
        Parameters
        ----------
        X,y : dataset for the model
        Returns
        -------
        None ;)
        """

        alphas = [(1e-5), 0.0001, 0.001, 0.003, 0.01,
                  0.03, 0.1, 0.25, 0.5, 1, 3, 10]
        grid_ridge = GridSearchCV(
            estimator=Ridge(), param_grid=dict(alpha=alphas))
        grid_lasso = GridSearchCV(
            estimator=Lasso(), param_grid=dict(alpha=alphas))

        grid_ridge.fit(X, y)
        grid_lasso.fit(X, y)

        print('\n\nUsing Grid Search ->')
        print("Ridge best alpha :", grid_ridge.best_estimator_.alpha)
        print("Ridge best coef :", np.hstack(
            (grid_ridge.best_estimator_.intercept_, grid_ridge.best_estimator_.coef_)))
        print("Lasso best alpha :", grid_lasso.best_estimator_.alpha)
        print("Lasso best coef :", np.hstack(
            (grid_lasso.best_estimator_.intercept_, grid_lasso.best_estimator_.coef_)))


if __name__ == '__main__':
    X, y, X_train, y_train, X_test, y_test = pre_process()
    regression = Linear_Regression()
    rmse_train, rmse_test, cost = regression.linearRegression(
        X_train, y_train, X_test, y_test)
    print("Linear Regression-\nRMSE at Training Dataset:%1.6f \nRMSE at Testing Dataset:%1.6f" %
          (rmse_train, rmse_test))
    regression.Ridge_lasso_Regression(X_train, y_train, X_test, y_test)
    regression.Grid_search_function(X, y)
    regression.plot_cost(cost)



