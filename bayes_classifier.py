from typing import List

__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"


class BayesClassifier:
    """Naive Bayes classifier implementation.

    """

    CUTOFF = lambda x: x >= 0.5

    def __init__(self, hypothesis: list, evidence: List[list]):
        self.hypothesis: list = hypothesis
        self.evidence: list = evidence
        # Prob of each effect given the hypothesis
        self.causal: dict = {}
        # Prob of hypothesis given each effect
        self.diagnostic: dict = {}

    def train(self):
        """Creating CPT for each node.

        :param data:
        :return:
        """
        # Calc causal/diagnostic
        for col in self.evidence:
            pass

    def train_causal(self, data: List[list]):
        for var in self.evidence:
            self.causal[var] = 0

    def train_diagnostic(self, data: List[list]):
        pass

    def test(self, data: List[list]) -> list:
        pass

    def predict(self):
        pass


class BayesNode:
    """Node in a naive Bayes' network.

    """

    def __init__(self, data: list, parents: list=None, children: list=None):
        self.parents: list = parents if parents is not None else []
        self.children: list = children if children is not None else []
        # self.cpt: dict = cpt if cpt is not None else {}
        self.data = data
        self.priors = {val: self.calc_prior(val) for val in set(self.data)}
        self.cpt = {'''do something'''} if len(children) > 0 else {}

    def calc_prior(self, val):
        """Calculates the prior probability for each evidence value."""
        count = 0
        for elem in self.data:
            count = count + 1 if val == elem else count
        return count / len(self.data)

    def set_parents(self, parents: list, update: bool=False) -> list:
        if update:
            self.parents.append(parents)
        else:
            self.parents = parents
        return self.parents

    def set_children(self, children: list, update: bool=False) -> list:
        if update:
            self.children.append(children)
        else:
            self.children = children
        return self.children
