from HW02 import *

def testPCA1():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
    print(data)
    # prepare the data
    x = data.iloc[:, 0:4]
    print(x)
    # Applying it to PCA function
    mat_reduced = PCA(x, 1)
    print(mat_reduced)


def testPCA2():
    data = pd.read_csv("data/stock.csv")
    print(data)
    # prepare the data
    x = data.iloc[:, 1:10]
    print(x)
    # Applying it to PCA function
    mat_reduced = PCA(x, 3)
    print(mat_reduced)

def main():
    print("This is HW02")
    #exa01()
    #exe02()
    testPCA2()
if __name__ == '__main__':
    main()
