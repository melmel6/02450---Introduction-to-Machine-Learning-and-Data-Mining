import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import model_selection
from toolbox_02450 import mcnemar


def error_fn(y_hat, y):
    error = np.sum(y_hat != y.squeeze()) / float(len(y))
    # error = (y_hat != y).mean()
    return error


# Create two Layer Cross Validation function
def twoLayerCrossValidation(model_types, parameter_types, X, y):
    # Set K-folded CV options
    random_seed = 1234  # Set random seed to ensure same results whenever CV is performed
    K1 = 10  # Number of inner loops
    K2 = 10  # Number of outer loops
    K1fold = model_selection.KFold(n_splits=K1, shuffle=True, random_state=random_seed)
    K2fold = model_selection.KFold(n_splits=K2, shuffle=True, random_state=random_seed)

    test_errors = np.zeros((K1, len(model_types) * 2))
    #  best_parameters = np.zeros((K1,len(model_types)))
    y_hat_list = []
    y_test_list = []
    for i in range(len(model_types)):
        y_hat_list.append([])

    # The two layer cross-validation algorithm
    for i, (par_index, test_index) in enumerate(K1fold.split(X, y)):
        print("Outer fold {} of {}".format(i + 1, K1))

        # Saves D_par and D_test to allow later statistical evaluation
        X_par = X[par_index, :]
        y_par = y[par_index]

        X_test = X[test_index, :]
        y_test = y[test_index]
        y_test_list.append(y_test)

        # Iterate over the three methods chosen for classification
        for m, (model_type, models) in enumerate(model_types):
            val_errors = np.zeros((K2, len(models)))

            # Inner cross validation loop
            for j, (train_index, val_index) in enumerate(K2fold.split(X_par, y_par)):
                X_train = X_par[train_index, :]
                y_train = y_par[train_index]

                X_val = X_par[val_index, :]
                y_val = y_par[val_index]

                # Test model type and calculate validation error for each model of the three methods
                for k, (name, parameter, model) in enumerate(models):
                    model.fit(X_train, y_train.squeeze())

                    y_hat = model.predict(X_val)
                    val_errors[j, k] = len(X_val) / len(X_par) * error_fn(y_hat, y_val)

            # Finds the optimal model
            inner_gen_errors = val_errors.sum(axis=0)
            best_model_index = np.argmin(inner_gen_errors)
            best_model_name, best_model_parameter, best_model = models[best_model_index]  # Determines optimal model

            if model_type == 'Base Line':
                model.fit(X_par, y_par.squeeze())
                y_hat = np.ones(len(y_test)) * model.predict(X_test)
            else:
                best_model.fit(X_par, y_par)
                y_hat = best_model.predict(X_test)
            y_hat_list[m].append(y_hat.squeeze())

            test_errors[i, m * 2 + 1] = error_fn(y_hat, y_test)  # Lists test_errors for each method and each outer fold
            test_errors[i, m * 2] = best_model_parameter  # List the best parameter type belonging to test-error

    test_errors_folds = pd.DataFrame.from_records(data=test_errors,
                                                  columns=sum([[parameter_types[i], model_types[i][0]] for i in
                                                               range(len(model_types))], []))

    return test_errors_folds, y_hat_list, y_test_list


# Baseline for classification
class BaseLine_Classification:
    def fit(self, X, y):
        self.bincount = np.bincount(y.astype(int)).argmax()

    def predict(self, X):
        return self.bincount


def ClassificationStatistics(test_errors, hats, tests):
    print(test_errors)
    predicted_log = np.concatenate(hats[0])
    predicted_KNN = np.concatenate(hats[1])
    predicted_BL = np.concatenate(hats[2])
    true_class = np.concatenate(tests)
    alpha = 0.05

    z_1, CI_log_vs_KNN, p_log_vs_KNN = mcnemar(true_class, predicted_log, predicted_KNN, alpha = 0.05)
    z_2, CI_log_vs_BL, p_log_vs_BL = mcnemar(true_class, predicted_log, predicted_BL, alpha = 0.05)
    z_3, CI_KNN_vs_BL, p_KNN_vs_BL = mcnemar(true_class, predicted_KNN, predicted_BL, alpha = 0.05)

    print("P_value for the null hypothesis: Log = KNN: ",p_log_vs_KNN)
    print(1-alpha, "% Confidence interval for difference in accuracy between log and KNN: ", CI_log_vs_KNN)
    print("")
    print("P_value for the null hypothesis: Log = BL: ",p_log_vs_BL)
    print(1-alpha, "% Confidence interval for difference in accuracy between log and BL: ", CI_log_vs_BL)
    print("")
    print("P_value for the null hypothesis: KNN = BL: ",p_KNN_vs_BL)
    print(1-alpha, "% Confidence interval for difference in accuracy between KNN and BL ", CI_KNN_vs_BL)

