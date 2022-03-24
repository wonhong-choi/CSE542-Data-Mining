from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from decisionTree import *
from SciKitTree import *


def testSciKitTree():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train,  y_train)
    tree.plot_tree(clf_gini)
    plt.show()
    clf_entropy = tarin_using_entropy(X_train, y_train)
    tree.plot_tree(clf_entropy)
    plt.show()
    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


def testDecisionTree():
    col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'type']
    data = pd.read_csv("wine.csv", skiprows=1, header=None, names=col_names)
    print(data.head(10))

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    print(X_train)

    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train,Y_train)
    #print('Root: ')
    #classifier.print_tree()

    Y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score

    print('\n\nAccuracy: ' + str(accuracy_score(Y_test, Y_pred)*100) + ' %')

def exa01():
    from sklearn.datasets import load_iris
    from sklearn import tree
    iris=load_iris()
    X, y= iris.data, iris.target
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(X,y)
    tree.plot_tree(clf)
    plt.show()



def main():
    #testSciKitTree()
    #testDecisionTree()
    exa01()


if __name__ == '__main__':
    main()