'''Maximilian Julius Lautenbach'''
import pickle
import numpy as np
from sklearn import svm

def pssm_predict(pssm_model, pssm_file):
    '''pssm predictor'''
    loaded_model = pickle.load(open(pssm_model, 'rb'))
    win = 25
    pad = win // 2
    pssm_extracted = np.genfromtxt(pssm_file, skip_header = 3, skip_footer = 5, usecols = range(22, 42))
    pssm_extracted_length = len(pssm_extracted)
    predicting_array = np.zeros((pssm_extracted_length, win, 20))
    decimal_pssm = pssm_extracted / 100
    pad_matrix = np.vstack([np.zeros((pad, 20)), decimal_pssm, np.zeros((pad, 20))])
    arrays = []
    for aa in range(pssm_extracted_length):
        predicting_array[aa] = pad_matrix[aa:aa + win]
    arrays.append(predicting_array.reshape(pssm_extracted_length, win *20))
    arrays = np.vstack(arrays)
#   prediction    
    results = loaded_model.predict(arrays)
    states = {1:'g', -1:'B'}
    results_decode = []
    for topology in results:
        results_decode.append(states[topology])
    string_result = ''.join(results_decode)
#
#
    with open('../output/pssm_svm_predictions.txt', 'w') as out_file:
        out_file.write(pssm_file[29:-11] + '\n')
        out_file.write(string_result + '\n')
        out_file.close()
    return pssm_file[29:-11] + '\n' + string_result
if __name__ == '__main__':
    print(pssm_predict('../output/pssm_model.sav', '../datasets/PSSM_files/PSSMs/>A5VZA8_PSEP1_observed_topology.fasta.pssm'))
