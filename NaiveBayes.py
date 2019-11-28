__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"

import argparse

from bayes_classifier import BayesClassifier


# Create parser and parse required CL args
parser = argparse.ArgumentParser()
parser.add_argument("TRAIN", nargs=1, type=str,
                    help="Training dataset in .csv format")
parser.add_argument("TEST", nargs=1, type=str,
                    help="Test dataset in .csv format")
parser.add_argument("MFILE", nargs=1, type=str,
                    help="Model file for saving model structure and the"
                         "probability distribution for each node")
parser.add_argument("RFILE", nargs=1, type=str,
                    help="Result file listing predicted/real value for each "
                         "observation along with the corresponding"
                         "confusion matrix")
args=parser.parse_args()
