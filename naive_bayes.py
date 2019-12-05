__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"

from collections import OrderedDict
from functools import reduce
import json
from typing import List, Tuple, Dict, Any, overload, Union


class NaiveBayes:

    def __init__(self, train: List[list], hyp: int, labels: List[str]=None):
        self.labels = self.label_dict(labels)
        self.hyp = hyp
        self.data, self.target = self.split_data(train, hyp)
        self.tdata = self.transpose(self.data)
        self.nodes: Dict[int, BayesNode] = {}
        self.hyp_prob = {c: self.target.count(c) / len(self.target)
                         for c in set(self.target)}

    # def prob(self, var: int, val: Any) -> float:
    #     prob = list(chain.from_iterable(
    #         filter(lambda r: [i for i in r if i == val], self.data)
    #     )).count(val) / len(self.data)
    #     return prob

    def prob(self, varval: Dict[int, Any]) -> float:
        """Calculates the probability each column (int) takes its mapped value.

        :param varval: dict of (column_number : value) mappings
        :return:
        """
        prob = len([row for row in self.data
                    if all([row[i] == varval[i]
                            for i in list(varval.keys())])]) / len(self.data)

        # for var, val in varval.items():
        #     probs[var] = list(chain.from_iterable(
        #         filter(lambda r: [i for i in r if i == val], self.data)
        #     )).count(val) / len(self.data)
        return prob

    def cond(self, var: int, val: Any, cond: Dict[int, Any]) -> float:
        bottom = self.prob(cond)
        cond.update({val: val})
        top = self.prob(cond)
        return top / bottom

    def train(self):
        for i in range(len(self.tdata)):
            self.nodes[i] = BayesNode(self.tdata[i], self.target)

    def test(self, test_data: List[list], hyp=None) -> Tuple[list, list]:
        target = self.target if hyp is None else hyp
        test_data, actual = self.split_data(test_data, target)
        predicted = []

        for row in test_data:
            max_cls, max_prob = 0, 0
            for val in set(self.target):
                product = reduce(lambda x,y: x*y,
                     [self.nodes[i].prob(row[i], val) for i in range(len(row))])
                product *= self.hyp_prob[val]
                if product > max_prob:
                    max_cls, max_prob = val, product
            predicted.append(max_cls)

        return predicted, target

    def get_distributions(self) -> str:
        distributions = OrderedDict()
        for i in range(len(self.labels)):
            if i < self.hyp:
                distributions.update({self.labels[i]: self.nodes[i]})
            elif i == self.hyp:
                distributions.update({self.labels[i]: self.nodes[i]})
            else:
                distributions.update({self.labels[i+1]: self.nodes[i+1]})
        return json.dumps(distributions)

    @classmethod
    def label_dict(cls, label_list) -> OrderedDict:
        labels = OrderedDict()
        for i in range(len(label_list)):
            labels[i] = label_list[i]
        return labels

    @classmethod
    def _nest_dict(cls, d: dict, keys: List[str], val: float) \
            -> Union[dict, float]:
        """Recursively constructs nested dicts with each key a word in an ngram.

        :param d: the current working dict
        :param keys: list of preceding words in ngram [c-1, ..., c-n+1]
        :param val: the probability of a given ngram
        :return: base case returns val; else, the dict constructed on unwinding
        """
        # if no more keys, at the end (beginning) of ngram; assign prob.
        if not keys:
            return val
        # word has been encountered before; pass its dict through to preserve
        elif keys[0] in d.keys():
            d[keys[0]] = cls._nest_dict(d[keys[0]], keys[1:], val)
            return d[keys[0]]
        # word has not been seen in this sequence; pass through blank dict
        else:
            d[keys[0]] = cls._nest_dict({}, keys[1:], val)
            return d[keys[0]]

    @classmethod
    def split_data(cls, data: List[list], target: int) \
            -> (List[list], List[list]):
        """Splits given 2D data matrix into non-target and target values.

        :param data: the dataset to be split
        :param target: the target column in the dataset
        :return: tuple of (2D list, list) for non-target and target values
        """
        if target < len(data):
            split = [list(r[:target+1])+list(r[target+2:]) for r in data]
        else:
            split = [r[:target+1] for r in data]
        target = [r[target] for r in data]
        return split, target

    @classmethod
    def transpose(cls, matrix: List[list]) -> List[list]:
        """Returns the transpose of a 2D list."""
        return list(map(list, zip(*matrix)))


class BayesNode:

    def __init__(self, evidence: list, hypothesis: list):
        self.ppt = {i: evidence.count(i) / len(evidence) for i in set(evidence)}
        self.cpt = {}
        self._train(evidence, hypothesis)

    def prob(self, value, given=None) -> float:
        if given is not None:
            return self.cpt[value][given]
        else:
            return self.ppt[value]

    def _train(self, evidence, hypothesis):
        self.cpt = {e: {h: 0 for h in set(hypothesis)} for e in set(evidence)}
        for row in map(tuple, zip(evidence, hypothesis)):
            self.cpt[row[0]][row[1]] += 1
        for e in set(evidence):
            for h in set(hypothesis):
                self.cpt[e][h] = self.cpt[e][h] / self.ppt[e]

    def get_distribution(self):
        return self.ppt, self.cpt
