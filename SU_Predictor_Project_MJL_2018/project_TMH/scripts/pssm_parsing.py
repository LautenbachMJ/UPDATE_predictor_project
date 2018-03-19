'''Maximilian Julius Lautenbach'''
#import sys 
#sys.path.insert(0,"../../codes") #if script is not in the same directory
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import os.path
#from svm_model_creator import data_input

def data_input(dataset):
    filehandle = open(dataset,'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    input_id = lines[0::3]
    input_seq = lines[1::3]
    input_top = lines[2::3]
        
    return input_id, input_seq, input_top 


def define_pssms(pssm_list, window):
    padding = window // 2
    arrays = []
    numbering = []
    
    for number, matrix in enumerate(pssm_list):
        length = len(matrix)
        training = np.zeros((length, window, 20))
        decimal_pssm = matrix / 100
        pad_matrix = np.vstack([np.zeros((padding, 20)), decimal_pssm, np.zeros((padding, 20))])
        for aa in range(length):
            training[aa] = pad_matrix[aa:aa + window]
            numbering.append(number)
        arrays.append(training.reshape(length, window *20))
    
    return np.vstack(arrays), numbering


def readin_pssm(window):
    '''Takes pssms from X_train and trains and saves model'''    
    
    input_id = []
    for file in os.listdir('../datasets/PSSM_files/PSSMs/'):
        pssm_list_train = [] 
        if file.endswith(".pssm"):
            ids = (os.path.join(file))
        input_id.append(ids)
    #return input_id    
        

    for ID in input_id:
        print(ID)
        pssm = os.path.join('../datasets/PSSM_files/PSSMs/' + ID) #+ #'.fasta.pssm' #all pssm locationssms
        pssm_list_train.append(np.genfromtxt(ids, skip_header=3, skip_footer=5, usecols=range(22,42)))
    #ERROR: file can't be found
    
    
    X_train = pssm_list_train
    
    
    X_train_changed, array_numbering = define_pssms(X_train, window) #X_train = pssm_list
    
    
    states = {'G':1, 'M':-1}
    Y_train_changed = []
    for proteins in input_top:    
        for topologies in proteins:
            y = states[topologies]
            Y_train_changed.append(y)

def pssm_train(x, y):
    '''svm training with x and y'''
    clf = svm.SVC(gamma=0.001, kernel='linear', C=1.0)
    clf.fit(x, y)
    filename = '../output/pssm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    
#    cvs_svm = cross_val_score(clf, seq, top, cv=cross_val, n_jobs=-1)
#    cvs_svm_mean = cvs_svm.mean()
#    clf.fit(x_train, y_train)
#    #prediction
#    y_test_top_predicted_svm = clf.predict(x_test)
#    svm_classreport = classification_report(y_test, y_test_top_predicted_svm, labels)
#    svm_confusionm = confusion_matrix(y_test, y_test_top_predicted_svm, labels)
#    svm_mcc = matthews_corrcoef(y_test, y_test_top_predicted_svm)
    
    
    return  
          
           
if __name__ == '__main__':  
    #input_id, input_seq, input_top  = data_input('../datasets/membrane-alpha_2state.3line_train_test.txt')
    print(readin_pssm(19))  
    #print(train_pssm(19)) #called with window size 19
