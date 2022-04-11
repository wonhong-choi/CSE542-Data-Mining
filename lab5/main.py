from regression import *
from GPEns import *
from RandomForest import *

def testLR():
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
              \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)


def testRF():
    # Training data
    mock_train = np.loadtxt("data/mock_data.csv", delimiter=",")
    mock_y = mock_train[ : , -1]

    # Build and train model
    rf1 = RandomForest(N_TREES,FOLD_SIZE)
    rf1.train(mock_train)
    rf1.print_trees()

    # Evaluate model on training data
    y_pred,y_pred_ind = rf1.predict(mock_train)
    print(f"Accuracy of random forest: {sum(y_pred == mock_y) / mock_y.shape[0]}")
    print("\nAccuracy for each individual tree:")
    c = 1
    for i in y_pred_ind:
        print("\nTree",c)
        print(f"Accuracy: {sum(i == mock_y) / mock_y.shape[0]}")
        c = c+1

def main():
    print("This is HW05")
    #testLR()
    #LR01()
    #MLR01()
    #DTR01()
    #testRF()
    GPExa01()

if __name__ == '__main__':
   main()