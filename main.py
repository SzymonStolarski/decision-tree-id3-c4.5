from components.szympans import Szympans
from components.classifiers.decision_tree_classifier\
    import DecisionTreeClassifier


DATA_PATH = 'data/car.data'
DLM = ','
DECISION_ATTRIBUTE_IDX = 6
CONDITIONAL_ATRIBUTES_IDX = [i for i in range(0, DECISION_ATTRIBUTE_IDX)]

data = Szympans.read_txt(path=DATA_PATH, dlm=DLM)

clf = DecisionTreeClassifier(
    conditional_attributes_idx=CONDITIONAL_ATRIBUTES_IDX,
    decision_attribute_idx=DECISION_ATTRIBUTE_IDX)

clf = DecisionTreeClassifier(CONDITIONAL_ATRIBUTES_IDX, DECISION_ATTRIBUTE_IDX)
clf.fit(data)

clf.visualize_tree
