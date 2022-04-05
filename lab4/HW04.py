#https://jakevdp.github.io/PythonDataScienceHandbook/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn import preprocessing

from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.svm import SVC

def NB01():
    X, y = make_blobs(1000, 2, centers=2, random_state=2, cluster_std=1.5)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Making the Confusion Matrix

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


    rng = np.random.RandomState(0)
    Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
    ynew = model.predict(Xnew)
    print(Xnew)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    lim = plt.axis()
    plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
    plt.axis(lim);
    plt.show()

    yprob = model.predict_proba(Xnew)
    yprob[-8:].round(2)
    print(yprob)

def NB02():
    # Importing the dataset
    dataset = pd.read_csv('data/Social_Network_Ads.csv')
    X = dataset.iloc[:, 0:2].values
    y = dataset.iloc[:, 2].values
    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the Naive Bayes model on the Training set

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix

    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def kNN01():
    df = pd.read_csv('data/iris-dataset.csv')
    print(df.head())

    X = df.iloc[:, 0:4]
    print(X)
    y = df.iloc[:, 4]
    print(y)

    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Train Model and Predict
    k = 5
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    Pred_y = neigh.predict(X_test)
    print(Pred_y)
    print("Accuracy of model at K=5 is", metrics.accuracy_score(y_test, Pred_y))

    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))
    plt.show()

    acc = []
    # Will take some time
    for i in range(1, 40):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, yhat))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), acc, color='blue', linestyle='dashed',
             marker='o', markerfacecolor='red', markersize=10)
    plt.title('accuracy vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
    plt.show()

def kNN02():
    iris = load_iris()
    X, y = iris.data[:, 2:], iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123,
                                                            shuffle=True)
    # plot dataset
    plt.scatter(X_train[y_train == 0, 0],
                X_train[y_train == 0, 1],
                marker='o',
                label='class 0 (Setosa)')

    plt.scatter(X_train[y_train == 1, 0],
                X_train[y_train == 1, 1],
                marker='^',
                label='class 1 (Versicolor)')

    plt.scatter(X_train[y_train == 2, 0],
                X_train[y_train == 2, 1],
                marker='s',
                label='class 2 (Virginica)')

    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')

    plt.show()

    #Fit k-Nearest Neighbor Model
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    #Use kNN Model to Make Predictions
    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('Test set accuracy: %.2f%%' % accuracy)

    # Visualize Decision Boundary
    plot_decision_regions(X_train, y_train, knn_model)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()

    plot_decision_regions(X_test, y_test, knn_model)
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


def svm01():
    #data
    X, y = make_blobs(n_samples=100, centers=2,
                      random_state=0, cluster_std=0.60)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
    plt.show()

    #linear discriminative classifier
    xfit = np.linspace(-1, 3.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    plt.xlim(-1, 3.5);

    plt.show()

    #Maximizing the Margin
    xfit = np.linspace(-1, 3.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                         color='#AAAAAA', alpha=0.4)

    plt.xlim(-1, 3.5);
    plt.show()
    #SVC
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model)

    plt.show()
    print(model.support_vectors_)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    for axi, N in zip(ax, [60, 120]):
        plot_svm(N, axi)
        axi.set_title('N = {0}'.format(N))
    plt.show()

def svm02():
    X, y = make_circles(100, factor=.1, noise=.1)

    clf = SVC(kernel='linear').fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf, plot_support=False);
    plt.show()
    r = np.exp(-(X ** 2).sum(1))
    print(r)
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()

    clf = SVC(kernel='rbf', C=1E6)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    plt.show()

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)



