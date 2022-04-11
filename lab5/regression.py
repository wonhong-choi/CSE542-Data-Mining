import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#https://www.geeksforgeeks.org/linear-regression-python-implementation/
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def LR01():
    dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print(dataset.head(5))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
    print(df)
    plt.scatter(X_test, y_test, color='red')
    plt.scatter(X_test, y_pred, color='green')
    plt.plot(X_train, regressor.predict(X_train), color='black')
    plt.title('Salary vs Experience (Result)')
    plt.xlabel('YearsExperience')
    plt.ylabel('Salary')
    plt.show()

def MLR01():
    dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/Startups_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print(dataset.head(5))

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
    print(df)



def DTR01():
    dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')
    X = dataset['Temperature'].values
    y = dataset['Revenue'].values

    print(dataset.head(5))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # Fitting Decision Tree Regression to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred = regressor.predict(X_test.reshape(-1, 1))

    df = pd.DataFrame({'Real Values': y_test.reshape(-1), 'Predicted Values': y_pred.reshape(-1)})
    print(df)
    # Visualising the Decision Tree Regression Results
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color='red')
    plt.scatter(X_test, y_pred, color='green')
    plt.title('Decision Tree Regression')
    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.show()

    plt.plot(X_grid, regressor.predict(X_grid), color='black')
    plt.title('Decision Tree Regression')
    plt.xlabel('Temperature')
    plt.ylabel('Revenue')
    plt.show()