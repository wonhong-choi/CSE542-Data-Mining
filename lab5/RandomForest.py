# https://github.com/aladdinpersson/Machine-Learning-Collection
from collections import Counter
import numpy as np
from random import randrange
from random import randint

# fold size (% of dataset size) e.g. 3 means 30%
FOLD_SIZE = 10
# number of trees
N_TREES = 20
# max tree depth
MAX_DEPTH = 30
# min size of tree node
MIN_NODE = 1


class RandomForest:
    def __init__(self, n_trees, fold_size):
        self.n_trees = n_trees
        self.fold_size = fold_size
        self.trees = list()

    """
        This function splits the given dataset into n-folds with replacement. The number of folds is equal to the number of the trees that will be trained.
        Each tree will have one fold as input. The size of the folds is a percentage (p) of the size of the original dataset. 
        Parameters:
        dataset: np array of the given dataset
        n_folds (int): number of folds in which the dataset should be split. Must be equal to the number of trees the user wants to train
        p (int): suggests the percentage of the dataset's size the size of a single fold should be.
        Returns list of np arrays: list with the k-folds 
    """

    def cross_validation_split(self, dataset, n_folds, p):
        dataset_split = list()
        fold_size = int(len(dataset) * p / 10)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset))
                fold.append(dataset[index])
            set = np.array(fold)
            dataset_split.append(set)
        return dataset_split

    """
        This function randomizes the selection of the features each tree will be trained on.
        Parameters:
            splits list of np arrays: list of folds

        Returns list of np arrays: list with the k-folds with some features randomly removed
    """

    def randomize_features(self, splits):
        dataset_split = list()
        l = len(splits[0][0])
        n_features = int((l - 1) * 5 / 10)
        for split in splits:
            for i in range(n_features):
                rng = list(range(len(split[0]) - 1))
                selected = rng.pop(randint(0, len(rng) - 1))
                split = np.delete(split, selected, 1)
            set = np.array(split)
            dataset_split.append(set)
        return dataset_split

    """
        Prints out all the decision trees of the random forest.

        BUG: The feature number is not representative of its initial enumeration in the original dataset due to the randomization. 
             This means that we do not know on which features each tree is trained on.
    """

    def print_trees(self):
        i = 1
        for t in self.trees:
            print("Tree#", i)
            temp = t.final_tree
            t.print_dt(temp)
            print("\n")
            i = i + 1

    """
        Iteratively train each decision tree.
        Parameters:
        X (np.array): Training data
    """

    def train(self, X):
        train_x = self.cross_validation_split(X, self.n_trees, self.fold_size)
        train_x = self.randomize_features(train_x)
        for fold in train_x:
            dt = DecisionTree(MAX_DEPTH, MIN_NODE)
            dt.train(fold)
            self.trees.append(dt)

    """
        This function outputs the class value for each instance of the given dataset as predicted by the random forest algorithm.
        Parameters:
        X (np.array): Dataset with labels
        Returns y (np.array): array with the predicted class values of the dataset
    """

    def predict(self, X):
        predicts = list()
        final_predicts = list()
        for tree in self.trees:
            predicts.append(tree.predict(X))
        # iterate through each tree's class prediction and find the most frequent for each instance
        for i in range(len(predicts[0])):
            values = list()
            for j in range(len(predicts)):
                values.append(predicts[j][i])
            final_predicts.append(max(set(values), key=values.count))
        return final_predicts, predicts

class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}

    """
        This function calculates the gini index of a split in the dataset
        Firstly the gini score is calculated for each child note and the resulting Gini is the weighted sum of gini_left and gini_right

        Parameters:
        child_nodes (list of np arrays): The two groups of instances resulting from the split

        Returns:
        float:Gini index of the split 

       """

    def calculate_gini(self, child_nodes):
        n = 0
        # Calculate number of all instances of the parent node
        for node in child_nodes:
            n = n + len(node)
        gini = 0
        # Calculate gini index for each child node
        for node in child_nodes:
            m = len(node)

            # Avoid division by zero if a child node is empty
            if m == 0:
                continue

            # Create a list with each instance's class value
            y = []
            for row in node:
                y.append(row[-1])

            # Count the frequency for each class value
            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2
            gini = gini + (m / n) * node_gini
        return gini

    """
            This function splits the dataset on certain value of a feature
            Parameters:
            feature_index (int): Index of selected feature

            threshold : Value of the feature split point


            Returns:
            np.array: Two new groups of split instances

           """

    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child

    """
                This function finds the best split on the dataset on each iteration of the algorithm by evaluating
                all possible splits and applying the one with the minimum Gini index.
                Parameters:
                data: Dataset

                Returns node (dict): Dictionary with the index of the splitting feature and its value and the two child nodes

               """

    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_score = 1000
        f_index = 0
        f_value = 0
        # Iterate through each feature and find minimum gini score
        for column in range(num_of_features):
            for row in data:
                value = row[column]
                l, r = self.apply_split(column, value, data)
                children = [l, r]
                score = self.calculate_gini(children)
                # print("Candidate split feature X{} < {} with Gini score {}".format(column,value,score))
                if score < gini_score:
                    gini_score = score
                    f_index = column
                    f_value = value
                    child_nodes = children
        # print("Chosen feature is {} and its value is {} with gini index {}".format(f_index,f_value,gini_score))
        node = {"feature": f_index, "value": f_value, "children": child_nodes}
        return node

    """
        This function calculates the most frequent class value in a group of instances
        Parameters:
        node: Group of instances

        Returns : Most common class value

    """

    def calc_class(self, node):
        # Create a list with each instance's class value
        y = []
        for row in node:
            y.append(row[-1])
        # Find most common class value
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]

    """
        Recursive function that builds the decision tree by applying split on every child node until they become terminal.
        Cases to terminate a node is: i.max depth of tree is reached ii.minimum size of node is not met iii.child node is empty
        Parameters:
        node: Group of instances
        depth (int): Current depth of the tree


    """

    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        # Check if tree has reached max depth
        if depth >= self.max_depth:
            # Terminate left child node
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            # Terminate right child node
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        # process left child
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        # process right child
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    """
        Apply the recursive split algorithm on the data in order to build the decision tree
        Parameters:
        X (np.array): Training data

        Returns tree (dict): The decision tree in the form of a dictionary.
    """

    def train(self, X):
        # Create initial node
        tree = self.find_best_split(X)
        # Generate the rest of the tree via recursion
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    """
        Prints out the decision tree.
        Parameters:
        tree (dict): Decision tree

    """

    def print_dt(self, tree, depth=0):
        if "feature" in tree:
            print(
                "\nSPLIT NODE: feature #{} < {} depth:{}\n".format(
                    tree["feature"], tree["value"], depth
                )
            )
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print(
                "TERMINAL NODE: class value:{} depth:{}".format(
                    tree["class_value"], tree["depth"]
                )
            )

    """
        This function outputs the class value of the instance given based on the decision tree created previously.
        Parameters:
        tree (dict): Decision tree
        instance(id np.array): Single instance of data

        Returns (float): predicted class value of the given instance
    """

    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    """
        This function outputs the class value for each instance of the given dataset.
        Parameters:
        X (np.array): Dataset with labels

        Returns y (np.array): array with the predicted class values of the dataset
    """

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)