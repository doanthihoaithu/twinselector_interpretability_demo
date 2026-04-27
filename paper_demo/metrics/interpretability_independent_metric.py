from paper_demo.metrics.InterpretabilityIndependentMetricAbstract import InterpretabilityIndependentMetricAbstract


class InterpretabilityScoreAverage(InterpretabilityIndependentMetricAbstract):
    @property
    def name(self) -> str:
        return "IndependentInterp"

    def score(self, y_true, interp_score):
        return interp_score[y_true == 1].mean()