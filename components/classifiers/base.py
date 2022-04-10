from abc import ABC, abstractmethod

from components.exceptions import ClassifierNotFitted


class BaseClassifier(ABC):

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_train_y_train: list[list]):
        # Implement logic here
        self._is_fitted = True
        return self

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise ClassifierNotFitted(
                f"This instance of {self.__class__.__name__} is not "
                f"yet fitted."
            )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
