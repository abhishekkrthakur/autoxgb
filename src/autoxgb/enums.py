import enum


class TaskType(enum.Enum):
    classification = 0
    regression = 1

    @staticmethod
    def from_str(task_type: str):
        if task_type == "classification":
            return TaskType.classification
        elif task_type == "regression":
            return TaskType.regression
        else:
            raise ValueError("Invalid task type: {}".format(task_type))

    @staticmethod
    def list_str():
        return ["classification", "regression"]


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
