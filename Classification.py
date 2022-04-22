from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from DataManipulation import *
from Functions import *

lambdas = np.arange(0.1, 50, 0.1)
model_types = [
    ("Logistic Regression", [
        ("Logistic Lambda = {}".format(l), l, LogisticRegression(solver='liblinear', C=1 / l))
        for l in lambdas
    ]),
    ("Nearest Neighbour", [
        ("{}NN".format(k), k, KNeighborsClassifier(n_neighbors=k))
        for k in np.arange(1, 20)
    ]),
    ("Base Line", [("Base Line", None, BaseLine_Classification())])

]

parameter_types = ["Lambda_vals", "Number of neighbours", "None"]  # The names of the parameters, for later convenience

test_errors, hats, tests = twoLayerCrossValidation(model_types, parameter_types, X, y)

ClassificationStatistics(test_errors, hats, tests)
