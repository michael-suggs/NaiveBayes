__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"

from collections import OrderedDict, defaultdict
from functools import reduce
import json
from typing import List, Tuple, Dict, Any, overload, Union


class NaiveBayes:
    """Naive Bayes net for predicting class (hyp) from all other variables.

    Attributes:
        labels:   Order-preserving dictionary mapping column number to its
            respective label.
        hyp:      Integer-index of the column we wish to predict (class column).
        data:     2D list of provided data with the class column removed.
        target:   The removed class column as a flat list.
        tdata:    Transposition of the data matrix; also a 2D list.
        nodes:    Dictionary mapping column number to BayesNode objects.
        hyp_prob: Probability of each value
    """

    def __init__(self, train: List[list], hyp: int, labels: List[str]=None):
        """Inits class attributes and calculates probabilities for hyp."""
        self.labels = self.label_dict(labels)
        self.hyp = hyp
        self.data, self.target = self.split_data(train, hyp)
        self.tdata = self.transpose(self.data)
        self.nodes: Dict[int, BayesNode] = {}
        self.hyp_prob = {c: self.target.count(c) / len(self.target)
                         for c in set(self.target)}
        self.prediction = OrderedDict()
        self.confusion = None

    # def prob(self, var: int, val: Any) -> float:
    #     prob = list(chain.from_iterable(
    #         filter(lambda r: [i for i in r if i == val], self.data)
    #     )).count(val) / len(self.data)
    #     return prob

    def prob(self, varval: Dict[int, Any]) -> float:
        """Calculates the probability each column (int) takes its mapped value.

        :param varval: dict of (column_number : column_value) mappings
        :return: probability that column_number will take column_value
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
        """Calculates prior/conditional probabilities for each variable."""
        for i in range(len(self.tdata)):
            self.nodes[i] = BayesNode(self.tdata[i], self.target, self.hyp_prob)

    def test(self, test_data: List[list], hyp=None) -> Tuple[list, list]:
        """Predicts class of each observation based on trained statistics."""
        target = self.hyp if hyp is None else hyp
        self.prediction.clear()
        test_data, actual = self.split_data(test_data, target)

        predicted = []
        row_len = len(test_data[0])

        for row in range(len(test_data)):
            max_cls, max_prob = 0, 0

            for val in set(self.target):
                product = reduce(lambda x,y: x*y,
                     [self.nodes[i].prob(test_data[row][i], val)
                      for i in range(row_len)])
                product *= self.hyp_prob[val]
                if product > max_prob:
                    max_cls, max_prob = val, product

            predicted.append(max_cls)
            self.prediction[row] = {
                'actual': actual[row],
                'predicted': max_cls,
                'probability': max_prob
            }

        self.confusion = self.confusion_matrix(predicted, actual)
        return predicted, actual

    def get_distributions(self) -> OrderedDict:
        """Returns the prior/conditional probabilities for each node."""
        distributions = OrderedDict()
        for i in range(len(self.labels)):
            if i < self.hyp:
                distributions.update({self.labels[i]: self.nodes[i]})
            elif i == self.hyp:
                distributions.update({self.labels[i]: self.nodes[i]})
            else:
                distributions.update({self.labels[i+1]: self.nodes[i+1]})
        return distributions

    def write_model(self, mfile: str):
        with open(mfile, 'w') as f:
            header = "{:10s} | {:10s} | {:10s}\n".format("LABEL", "GIVEN",
                                                         "DISTRIBUTION")
            hyp_label = self.labels[self.hyp]
            f.write(header)
            f.write("=" * (len(header) - 1) + "\n")
            for label, node in self.get_distributions().items():
                ppt, cpt = node.tabulate_distribution()
                f.write("{: <10s} | {: <10s} | {: >10s}\n".format(label, hyp_label, "problty."))
                f.write(str("="*36 + "\n"))
                for row in ppt:
                    f.write("{: <10s} | {: <10s} | {: >10f}\n".format(
                        row[0], "", row[1]))
                for row in cpt:
                    f.write("{: <10s} | {: <10s} | {: >10f}\n".format(
                        row[0], row[1], row[2]))

    def write_results(self, rfile: str):
        with open(rfile, 'w') as f:
            header = "{:10s} | {:10s} | {:12s}".format(
                'ACTUAL', 'PREDICTED', 'PROBABILITY')
            f.write(header + "\n")
            f.write("="*len(header) + "\n")
            for key, val in self.prediction.items():
                f.write("{:10s} | {:10s} | {:12f}\n".format(
                    val['actual'],
                    val['predicted'],
                    val['probability']
                ))

            f.write("\n\nCONFUSION MATRIX\n\n")
            for k, v in self.confusion:
                f.write("{: <5s} : {: >5s}\n".format(k, v))

    @classmethod
    def confusion_matrix(cls, predicted, actual) -> Dict[str, int]:
        """Generates dict counting the number of TP/TN/FP/FN predictions."""
        matrix: Dict[str, int] = defaultdict(int)
        for i in range(len(predicted)):
            pred, act = predicted[i], actual[i]
            if (pred == 1) and (act == 1):
                matrix['TP'] += 1
            elif (predicted == 0) and (act == 0):
                matrix['TN'] += 1
            elif predicted[i] > actual[i]:
                matrix['FP'] += 1
            elif predicted[i] < actual[i]:
                matrix['FN'] += 1
        return matrix

    @classmethod
    def label_dict(cls, label_list) -> OrderedDict:
        """Generates order-preserving dict mapping labels to columns."""
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
    """Naive Bayes node, representing a single variable (column) in the data.

    Attributes:
        ppt:    The prior probability table for the represented variable.
        cpt:    The conditional probability table for the represented variable.
    """

    def __init__(self, evidence: list, hypothesis: list, hprob: dict):
        """Initiates variables, calculates priors, and trains the node."""
        self.ppt = {i: evidence.count(i) / len(evidence) for i in set(evidence)}
        self.cpt = {}
        self._train(evidence, hypothesis, hprob)

    def prob(self, value, given=None) -> float:
        """Returns probability that this variable will take value `value`.

        If `given` is provided, returns the probability that this variable will
        take `value` as its value provided that we know the hypothesis has
        taken value `given`. As this is a naive network, the only possible
        parent is that of the hypothesis node and therefore we need not consider
        any other nodes/variables in the network.

        :param value: the variable value to retrieve the probability of
        :param given: if present, calculates P( value | given )
        :return:
        """
        if given is not None:
            return self.cpt[value][given]
        else:
            return self.ppt[value]

    def _train(self, evidence, hypothesis, hprob) -> None:
        """Generates the conditional probability table for this node.

        :param evidence: value list for this variable in order of observation
        :param hypothesis: value list for the hypothesis in order of observation
        :return:
        """
        self.cpt = {e: {h: 0 for h in set(hypothesis)} for e in set(evidence)}
        for row in map(tuple, zip(evidence, hypothesis)):
            self.cpt[row[0]][row[1]] += 1
        for e in set(evidence):
            for h in set(hypothesis):
                self.cpt[e][h] = self.cpt[e][h] / evidence.count(e)

    def get_distribution(self) -> Tuple[dict, dict]:
        """Returns the prior/conditional probability table of this node."""
        return self.ppt, self.cpt

    def tabulate_distribution(self):
        """Returns tabular format for probability distributions."""
        prior_table = [(val, prob) for val, prob in self.ppt.items()]
        cond_table = [(node, hyp, prob) for node in self.cpt.keys()
                      for hyp, prob in self.cpt[node].items()]
        return prior_table, cond_table
