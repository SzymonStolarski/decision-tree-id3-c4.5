from __future__ import annotations
import math

from components.node import Node
from components.szympans import Szympans
from components.classifiers.base import BaseClassifier


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision tree classifier.
    """
    def __init__(self, conditional_attributes_idx: list[int],
                 decision_attribute_idx: int) -> None:
        super().__init__()
        self.conditional_attributes_idx = conditional_attributes_idx
        self.decision_attribute_idx = decision_attribute_idx
        self.__fitted_tree = None

    def fit(self, X_train_y_train: list[list]):
        self.__fitted_tree = self.__build_tree(X_train_y_train)
        self._is_fitted = True

        return self

    def __build_tree(self, X_train_y_train: list[list],
                     branch: str = None, test: str = None) -> Node:
        node = Node()
        calc_gain_ratio = self.calculate_gain_ratio(X_train_y_train)
        selected_attribute = self.select_attribute(calc_gain_ratio)

        # Recursive tree building
        if not sum([gain for gain in calc_gain_ratio.values()]) == 0:
            node.test = selected_attribute
            children_data = self.get_children_data(X_train_y_train, node.test)

            for child in children_data:
                child_node = self.__build_tree(
                    child, child[0][selected_attribute], node.test)
                node.children.append(child_node)

        else:
            node.leaf = X_train_y_train[0][self.decision_attribute_idx]
            branch = X_train_y_train[0][test]

        node.branch = branch

        return node

    def calculate_entropy(self, data: dict) -> float:
        p = [(value/sum(data.values())) for value in data.values()]
        entropy = -sum([value*math.log2(value) for value in p])

        if entropy == -0.0:
            entropy *= -1

        return entropy

    def calculate_gain_ratio(self, data: list[list]) -> dict:

        info_in_all_attributes = {}
        gain_in_all_attributes = {}
        gain_ratio_in_all_attributes = {}

        previous_information = self.calculate_entropy(
                               Szympans.count_values(data)[
                                   self.decision_attribute_idx])
        for conditional_attr in self.conditional_attributes_idx:

            occurences_in_attribute = Szympans.count_values(
                                      data)[conditional_attr]
            no_of_all_obs = len(data)

            info_in_attribute = 0
            for j in occurences_in_attribute.keys():

                filtered_by_condidional_attribute = [
                    observation for observation
                    in data if observation[conditional_attr] == j]

                partial_info = (
                    occurences_in_attribute[j]/no_of_all_obs) *\
                    self.calculate_entropy(
                        Szympans.count_values(
                            filtered_by_condidional_attribute
                            )[self.decision_attribute_idx])

                info_in_attribute += partial_info

            gain_in_attribute = previous_information - info_in_attribute

            split_info_in_attribute = self.calculate_entropy(
                Szympans.count_values(data)[conditional_attr])

            try:
                gain_ratio_in_attribute = \
                    gain_in_attribute/split_info_in_attribute
            except ZeroDivisionError:
                gain_ratio_in_attribute = 0

            info_in_all_attributes[conditional_attr] = info_in_attribute
            gain_in_all_attributes[conditional_attr] = gain_in_attribute
            gain_ratio_in_all_attributes[
                conditional_attr] = gain_ratio_in_attribute

        return gain_ratio_in_all_attributes

    def select_attribute(self, metric_dict: dict) -> int:
        return max(metric_dict, key=metric_dict.get)

    def get_children_data(self, parent_data: list[list],
                          selected_attribute: int) -> list[list]:

        children_data = []
        unique_values = Szympans.count_values(
            parent_data)[selected_attribute].keys()

        for unique_value in unique_values:
            children_data.append(
                [row for row in parent_data
                 if row[selected_attribute] == unique_value])

        return children_data

    @property
    def fitted_tree(self) -> Node:
        return self.__fitted_tree

    @property
    def visualiza_tree(self) -> None:
        """
        Recursive tree visualization.
        """
        def visualize_tree_recursive(fitted_tree: Node, shift: int) -> None:
            if fitted_tree.test is not None:
                print("-" * shift, end="")
                if fitted_tree.branch:
                    print(fitted_tree.branch, end=" -> ")
                print("Atrybut:", fitted_tree.test)
                for child in fitted_tree.children:
                    visualize_tree_recursive(child, shift+4)
            else:
                print("-" * shift, end="")
                print(fitted_tree.branch, "->", fitted_tree.leaf)

        visualize_tree_recursive(self.__fitted_tree, shift=0)
