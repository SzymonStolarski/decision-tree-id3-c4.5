import math

from components.szympans import Szympans


class DecisionTreeCalculator:
    """
    Decision tree builder. Work still in progress.
    """
    def __init__(self) -> None:
        pass

    def calculate_entropy(self, data: dict) -> float:
        p = [(value/sum(data.values())) for value in data.values()]
        entropy = -sum([value*math.log2(value) for value in p])

        entropy *= -1 if entropy == -0.0 else entropy

        return entropy

    def calculate_information(self, data, conditional_attributes: list[int],
                              decision_attribute: int) -> tuple:

        info_in_all_attributes = {}
        gain_in_all_attributes = {}
        gain_ratio_in_all_attributes = {}

        previous_information = self.calculate_entropy(
                               Szympans.count_values(data)[decision_attribute])
        for conditional_attr in conditional_attributes:

            occurences_in_attribute = Szympans.count_values(
                                      data)[conditional_attr]
            no_of_all_obs = len(data)

            info_in_attribute = 0
            for j in occurences_in_attribute.keys():

                filtered_by_condidional_attribute = [observation for observation 
                                                     in data if observation[conditional_attr]==j]

                partial_info = (occurences_in_attribute[j]/no_of_all_obs) * self.calculate_entropy(
                                Szympans.count_values(filtered_by_condidional_attribute)[decision_attribute])

                info_in_attribute += partial_info

            gain_in_attribute = previous_information - info_in_attribute

            split_info_in_attribute = self.calculate_entropy(Szympans.count_values(data)[conditional_attr])
            gain_ratio_in_attribute = gain_in_attribute/split_info_in_attribute

            info_in_all_attributes[conditional_attr] = info_in_attribute
            gain_in_all_attributes[conditional_attr] = gain_in_attribute
            gain_ratio_in_all_attributes[conditional_attr] = gain_ratio_in_attribute

        return info_in_all_attributes, gain_in_all_attributes, gain_ratio_in_all_attributes

    def select_attribute(self, metric_dict):
        return max(metric_dict, key=metric_dict.get)
