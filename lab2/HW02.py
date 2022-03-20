import numpy as np
import pandas as pd

def exa01():
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
                                                    'h'], columns=['one', 'two', 'three'])
    df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    print("Dataset:")
    print(df)

    print("Data Cleaning.....")
    print(df.dropna())
    print(df.fillna(df.mean()))
    print(print(df.dropna(axis=1)))

    #Count missing data objects
    print("Count missing data objects")
    print("Missing values per column:")
    print(df.apply(num_missing, axis=0))
    print("\nMissing values per row:")
    print(df.apply(num_missing, axis=1))

    #Remove Dulpicate rows
    print("Remove Dulpicate rows")
    df['is_duplicated'] = df.duplicated(['one'])
    print("Dataset showing Duplicate rows\n", df)
    print("\nTotal duplicate rows:", df['is_duplicated'].sum())
    df_nodup = df.loc[df['is_duplicated'] == False]
    print("\nDataset after removing duplicate rows:\n", df_nodup)

def num_missing(x):
    return sum(x.isnull())

def exe02():
    stocks_df = pd.read_csv("data/stock.csv")
    print(stocks_df.describe())
    for i in stocks_df.columns[1:]:
        print(i)
    print(stocks_df.isnull().sum())
    X=stocks_df.drop(columns = ['Date'])
    print(X)
    print(X.corr())
    print(X/X.max())
    print((X - X.min()) / (X.max() - X.min()))
    print((X-X.mean())/X.std())
    [X[col].update((X[col] - X[col].min()) / (X[col].max() - X[col].min())) for col in X.columns]
    print(X)


def PCA(X, num_components):
    # Step-1 Subtract the mean of each variable
    X_mean=np.mean(X, axis=0)
    print('mean vector')
    print(X_mean)
    X_meaned = X - X_mean
    print('mean subtracted data')
    print(X_meaned)


    # Step-2  Calculate the Covariance Matrix
    cov_mat = np.cov(X_meaned, rowvar=False)
    print('Covariance Matrix')
    print(cov_mat)

    # Step-3 Compute the Eigenvalues and Eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    print('eigen_values')
    print(eigen_values)

    print('eigen_vectors')
    print(eigen_vectors)


    # Step-4 Sort Eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    print('sorted_eigenvalue')
    print(sorted_eigenvalue)

    print('sorted_eigenvectors')
    print(sorted_eigenvectors)

    # Step-5 Select a subset from the rearranged Eigenvalue matrix
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6  Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    print('Transformed data')
    print(X_reduced)
    return X_reduced

