'''thank you lucie '''
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from svm_model_creator_optimisation import one_hot, top_encoding
def gridscore(dataset):
    '''grid search to find better parameters - coding help from lucie'''
    for window in range(17, 21, 2):
        aa_array = one_hot(dataset, window)
        top_array = top_encoding(dataset, window)
        x_train, y_train = aa_array, top_array
        c_range = [1, 10, 2]
        g_range = [0.001, 0.1, 0.01]
        param = {'C' : c_range, 'gamma' : g_range}
        clf = GridSearchCV(SVC(), param, n_jobs=-1, cv=5, verbose=True,
                           error_score=np.NaN, return_train_score=False)
        clf.fit(x_train, y_train)
        datafile = pd.DataFrame(clf.cv_results_)
        filename = '../output/GRID_SEARCH/' + str(window) + '_gridscored_on_50' + '.csv'
        datafile.to_csv(filename, sep='\t', encoding='UTF-8') #tab sep
if __name__ == '__main__':
    gridscore('../datasets/membrane-alpha_2state.3line_50.txt')
