from abc import ABC, abstractmethod


class AccuracyMetricAbstract(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def score(self, y_true, y_score):
        pass

