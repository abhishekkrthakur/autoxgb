import enum


class ProblemType(enum.IntEnum):
    binary_classification = 1
    multi_class_classification = 2
    multi_label_classification = 3
    single_column_regression = 4
    multi_column_regression = 5

    @staticmethod
    def from_str(label):
        if label == "binary_classification":
            return ProblemType.binary_classification
        elif label == "multi_class_classification":
            return ProblemType.multi_class_classification
        elif label == "multi_label_classification":
            return ProblemType.multi_label_classification
        elif label == "single_column_regression":
            return ProblemType.single_column_regression
        elif label == "multi_column_regression":
            return ProblemType.multi_column_regression
        else:
            raise NotImplementedError
