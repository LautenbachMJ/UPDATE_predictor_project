'''from maria'''
#import sys 
#sys.path.insert(0,"../../codes") #if script is not in the same directory
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
#from svm_model_creator import data_input

def data_input(dataset):
    filehandle = open(dataset,'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    input_id = lines[0::3]
    input_seq = lines[1::3]
    input_top = lines[2::3]
        
    return input_id, input_seq, input_top 


def extract_pssms(pssm_list, window):
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


def train_pssm(input_id, input_top, win, out_file_path):
    '''Takes pssms from X_train and trains and saves model'''    
    
    pssm_list_train = []    
    for ID in input_id:
        pssm = '../datasets/PSSM_files/PSSMs/' + ID + '.fasta.pssm' #location of your pssms
    
        pssm_list_train.append(np.genfromtxt(pssm, skip_header=3, skip_footer=5, usecols=range(22,42)))
    X_train = pssm_list_train
    
    ###################################################
    
    ###################################################
    X_train_changed, array_numbering = extract_pssms(X_train, win) #X_train = pssm_list
    
    
    states = {'g':1, 'B':-1}
    Y_train_changed = []
    for proteins in input_top:    
        for topologies in proteins:
            y = states[topologies]
            Y_train_changed.append(y)


    x_train, x_test, y_train, y_test = train_test_split(X_train_changed, Y_train_changed, test_size=0.33, random_state=42)
    seq = x_train
    top = y_train
    cross_val = 5
    labels = [1, -1]

#   training
    clf = svm.SVC(gamma=0.001, kernel='linear', C=1.0) 

    filename = '../output/pssm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    cvs_svm = cross_val_score(clf, seq, top, cv=cross_val, n_jobs=-1)
    cvs_svm_mean = cvs_svm.mean()
    clf.fit(x_train, y_train)
#   prediction
    y_test_top_predicted_svm = clf.predict(x_test)
    svm_classreport = classification_report(y_test, y_test_top_predicted_svm, labels)
    svm_confusionm = confusion_matrix(y_test, y_test_top_predicted_svm, labels)
    svm_mcc = matthews_corrcoef(y_test, y_test_top_predicted_svm)
    
    with open(out_file_path + str(win) + 'winsize_pssm_based_model_scoringresults.txt', 'w') as out_file:
        out_file.write('Cross-validation scores for PSSM-SVC: ' + str(cvs_svm_mean)+ '\n')
        out_file.write('Matthews correlation coefficient (MCC) SVM: ' + str(svm_mcc) + '\n')
        out_file.write('Classification report SVM: ' + '\n' + str(svm_classreport) + '\n')
        out_file.write('Confusion matrix SVM: ' + '\n' + str(svm_confusionm) + '\n')
    out_file.close()
    
    print(svm_classreport)
    print(svm_confusionm)
    print(svm_mcc)
    
    return  
            
           
if __name__ == '__main__':  
    input_id, input_seq, input_top  = data_input('../datasets/membrane-beta_2state.3line_10.txt')
    train_pssm(input_id, input_top, 25, '../output/WIN_ALGO_OPTI/')
