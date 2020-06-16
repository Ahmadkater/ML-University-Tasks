import numpy as np
import pandas as pd
import operator


class TreeNode:

    def __init__(self, feature=None, right_child=None, left_child=None, threshold=None, *, value=None):

        self.feature = feature
        self.right_child = right_child
        self.left_child = left_child
        self.threshold = threshold
        self.value = value

    def check_leaf_node(self):
        return self.value is not None


class DTree:

    def __init__(self, max_tree_depth=20, splits_min=2):

        self.root = None

        self.splits_min = splits_min
        self.max_depth = max_tree_depth

    def build_tree(self, features_matrix, labels_vector):

        self.root = self.grow(features_matrix, labels_vector)

    def grow(self, features_matrix, labels_vector,depth=0):

        records_num = len(features_matrix)
        features_num = len(features_matrix.columns)
        labels_count = len(labels_vector.unique())

        # Stop when there is a leaf node

        if depth >= self.max_depth or labels_count < 2 or records_num < self.splits_min:

            leaf_node_value = self.get_common_label(labels_vector)

            return TreeNode(value=leaf_node_value)

        # greedy search

        best_feature, best_threshold = self.get_best(features_matrix,labels_vector)
        left_indices, right_indices = self.branch(features_matrix[best_feature],best_threshold)
        features_matrix = features_matrix.drop(best_feature,axis=1)
        left_branch = self.grow(features_matrix.loc[left_indices], labels_vector[left_indices], depth+1)
        right_branch = self.grow(features_matrix.loc[right_indices], labels_vector[right_indices], depth+1)

        return TreeNode(best_feature,right_branch,left_branch,best_threshold)

    def get_best(self, features_matrix, labels_vector):

        best_gain = -1
        split_feature, split_threshold = None, None

        for feature_vector in features_matrix.columns:
            vector = features_matrix[feature_vector]
            thresholds = vector.unique()
            for t in thresholds:
                gain = self.calculate_information_gain(vector,labels_vector,t)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_vector
                    split_threshold = t

        return split_feature, split_threshold

    def calculate_information_gain(self,feature_vector,labels,splitting_threshold):

        parent_entropy = entropy(labels)

        left, right = self.branch(feature_vector,splitting_threshold)

        total = len(labels)
        left_count = len(left)
        right_count = len(right)

        if left_count == 0 or right_count == 0:
            return 0

        left_entropy = entropy(labels[left])
        right_entropy = entropy(labels[right])

        child_entropy = (left_count / total) * left_entropy + (right_count / total) * right_entropy

        return parent_entropy - child_entropy

    def branch(self,fv,threshold):

        left = fv[fv <= threshold ].index.to_list()
        right = fv[fv > threshold ].index.to_list()

        return left,right

    def get_common_label(self, labels_vector):

        categories_count = dict(labels_vector.value_counts())
        common_label = max(categories_count.items(), key=operator.itemgetter(1))[0]

        return common_label

    def prediction(self, records):

        predictions = []

        for index, row in records.iterrows():
            record = pd.DataFrame(columns=records.columns)

            record.loc[index] = row

            value = self.traverse_tree(record, self.root)

            predictions.append(value)

            del record

        return np.array(predictions)

    def traverse_tree(self,record, node):

        if node.check_leaf_node():

            return node.value

        if record[node.feature].values[0] <= node.threshold:

            return self.traverse_tree(record, node.left_child)

        return self.traverse_tree(record, node.right_child)


def entropy(vector):

    categories = vector.value_counts().to_numpy()
    total = np.sum(categories)
    ps = [p/total for p in categories]

    return -np.sum([p * np.log2(p) for p in ps if p > 0])

