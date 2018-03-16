print(__doc__)
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from svm_model_creator_optimisation import one_hot, top_encoding

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from sklearn.model_selection import validation_curve

from sklearn.model_selection import GridSearchCV
from svm_model_creator_optimisation import one_hot, top_encoding



def gridscore(dataset):
#    '''grid search to find better parameters - coding help from lucie'''
#    for window in range(21, 25, 2):
##        aa_array = one_hot(dataset, window)
#        top_array = top_encoding(dataset, window)
#        x_train, y_train = aa_array, top_array
#        c_range = [1, 10, 2]
#        g_range = [0.001, 0.01]
#        param = {'C' : c_range, 'gamma' : g_range}
#        clf = GridSearchCV(SVC(), param, n_jobs=-1, cv=3, verbose=True,
##                           error_score=np.NaN, return_train_score=False)
#        clf.fit(x_train, y_train)
##       datafile = pd.DataFrame(clf.cv_results_)
#        filename = '../output/GRID_SEARCH/' + str(window) + '_gridscored' + '.csv'
#        datafile.to_csv(filename, sep='\t', encoding='UTF-8') #tab sep
#
    win = 3
    X = one_hot(dataset, win)
    y = top_encoding(dataset, win)

    param_range = [1, 10, 1]
    train_scores, test_scores = validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    gridscore('../datasets/membrane-beta_2state.3line_10.txt')
