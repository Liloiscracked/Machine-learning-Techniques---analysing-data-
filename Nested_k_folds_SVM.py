#Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
##############################
##############################
def model_selection(list_c, X_train, Y_train, scoring='accuracy', k_fold=5):
    """
    The function aims at selecting the optimal hyper-parameter(s) values.
    It performs an inner cross validation
    """
    list_accs = []
    for c in list_c:
        clf = svm.SVC(kernel='linear', C=c)
        acc = np.average(cross_val_score(clf, X_train, Y_train, cv=k_fold, scoring=scoring))
        list_accs.append(acc)
    return list_c[list_accs.index(max(list_accs))]

def nested_cross_validation(list_k_neighbors, X, Y, scoring='accuracy', k_fold=5):
    raw_indices = np.array(range(Y.shape[0]))
    kf = KFold(n_splits=k_fold)
    kf.get_n_splits(raw_indices)

    list_accs = []
    # Outer cross validation
    for fold_idx, (train_index, test_index) in enumerate(kf.split(raw_indices)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        c_optimal = model_selection(list_k_neighbors, X_train, Y_train, scoring=scoring, k_fold = k_fold)
        clf = svm.SVC(kernel='linear', C=c_optimal)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print("Fold %d: Optimal C: %d, Accuracy: %.4f" % (fold_idx, c_optimal, acc))
        list_accs.append(acc)
    return list_accs
##############################
##############################
# Load dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
Y = cancer.target

raw_indices = np.array(range(Y.shape[0]))
kf = StratifiedKFold(n_splits=5, shuffle=True)
kf.get_n_splits(raw_indices)
# Defing list of values for C hyper-parameter of SVM
list_c = [10**(-3), 10**(-2), 10**(-1), 1, 100, 1000]
k_fold = 5

list_accs = nested_cross_validation(list_c, X, Y, scoring='accuracy', k_fold=k_fold)
print("Average Accuracy: ", np.average(list_accs))