from abc import ABC, abstractmethod


class InterpretabilityIndependentMetricAbstract(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, y_true, interp_score):
        pass

