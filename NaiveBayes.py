__author__ = "Michael J. Suggs // mjsuggs@ncsu.edu"

import argparse, csv

from naive_bayes import NaiveBayes


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
args = parser.parse_args()


def run_network(train, test, headers):
    hypothesis = headers.index('class') if 'class' in headers else -1
    bayes_net = NaiveBayes(train, hypothesis, headers)
    bayes_net.train()
    predicted, actual = bayes_net.test(test)
    bayes_net.write_model(args.MFILE[0])
    bayes_net.write_results(args.RFILE[0])


def load_data(fname):
    with open(fname, 'r') as csvf:
        csvr = csv.reader(csvf)
        headers = next(csvr, None)
        data = [list(row) for row in csvr]
    return data, headers


if __name__ == '__main__':
    # print("{}, {}, {}, {} : ".format(args.TRAIN, args.TEST, args.MFILE, args.RFILE))
    mr_train, headers = load_data(args.TRAIN[0])
    mr_test, _ = load_data(args.TEST[0])

    run_network(mr_train, mr_test, headers)