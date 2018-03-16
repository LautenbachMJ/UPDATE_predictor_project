'''Maximilian Julius Lautenbach'''
import pickle
import numpy as np
#from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
############################################################################
#=============================== parser ===================================#
############################################################################
def data_input(dataset):
    '''pareses input data'''
    filehandle = open(dataset, 'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    prot_id = lines[0::3]
    prot_seq = lines[1::3]
    prot_top = lines[2::3]
    return prot_seq, prot_top, prot_id
# ======================== sliding window ==================================
def window(dataset, win):
    '''creates windows in sequence'''
    seq = data_input(dataset)[0]
    # win = 21 #literature window size (TMSEG) but 31 seems to be a good window size too
    pad = win//2 #equals to two
    seq_win_list = []
    temp = []
    for element in seq:
        char = ((pad)*'0' + element + (pad)*'0')
        temp.append(char)
    for padded_seq in temp:
        tempo = []
        for residue  in range(len(padded_seq)):
            if residue+win > len(padded_seq):
                break
            tempo.append(padded_seq[residue:residue+win])
        seq_win_list.append(tempo)
    return seq_win_list, win
# ========================= aa into num ====================================
def a_to_num(dataset, win):
    '''transforms aa into numbers'''
    seq_data = window(dataset, win)[0]
    aa_map = {'K': 12, 'Q': 7, 'I': 10, 'W': 18, 'T': 17, 'R': 2, 'V': 20,
              'E': 6, '0': 0, 'H': 9, 'P': 15, 'G': 8, 'N': 3, 'A': 1,
              'Y': 19, 'L': 11, 'F': 14, 'D': 4, 'M': 13, 'S': 16, 'C': 5}
    seq_num = []
    for sequence in seq_data:
        temp = []
        for windows in sequence:
            encoded = [aa_map[aa] for aa in windows]
            temp.append(encoded)
        seq_num.append(temp)
    return seq_num
# ======================== num into binary =================================
def one_hot(dataset, win):
    '''encodes sequence numbers into binary'''
    one_hot_input = a_to_num(dataset, win)
    aa_total = '0ARNDCEQGHILKMFPSTWYV'
    aa_onehot_encoded = []
    for protein in one_hot_input:
        this_prot = []
        for win in protein:
            this_win = []
            for residue in win:
                letter = [0 for _ in range(len(aa_total))]
                if residue != 0:
                    letter[residue] = 1
                this_win.extend(letter)
            this_prot.append(this_win)
        aa_onehot_encoded.extend(this_prot)
    aa_onehot_encoded = np.array(aa_onehot_encoded)
    return aa_onehot_encoded
#================== topology into num ======================================
def top_encoding(dataset, win):
    '''encodes topology into numbers(binary in the case of 2 states)'''
    top_data = data_input(dataset)[1]
    top = 'GM'
    top_char_to_int = dict((c, i) for i, c in enumerate(top))
    top_num = []
    for proteins in top_data:
        temp = []
        for tops in proteins:
            residue_top = top_char_to_int[tops]
            temp.append(residue_top)
        top_num.extend(temp)
    return top_num
#========== saving encoded sequences & topologies into np.file =============
def np_savez(dataset, win, svm_input):
    '''saves the model as a np file'''
    aa_list = one_hot(dataset, win)
    top_list = top_encoding(dataset, win)
    np.savez(svm_input, x=aa_list, y=top_list)
    return aa_list, top_list
############################################################################
#=========================== SVM Training =================================#
############################################################################
def prediction_fct(dataset, prediction_model, input_file_array, win, out_file_path):
    '''accuracy test on 50 UNTOUCHED proteins'''
    np.set_printoptions(threshold=np.inf)
    
#    x_train, x_test, y_train, y_test = train_test_split(loaded['x'], loaded['y'], test_size=0.33, random_state=42)

    #prediction model loading
    loaded_model = pickle.load(open(prediction_model, 'rb'))
    
    #50UNTOUCHED loading
    loaded = np.load(input_file_array)
    x, y = loaded['x'], loaded['y']
    #print(x)
    #print(y)
    
    
    input_id = data_input(dataset)[2]
    input_seq = data_input(dataset)[0]
    input_top = data_input(dataset)[1]
    
    #prediction
    top_y_predicted = loaded_model.predict(x)
    
    svm_classreport = classification_report(y, top_y_predicted, labels = [0, 1])
    svm_confusionm = confusion_matrix(y, top_y_predicted, labels = [0, 1])
    
    svm_mcc = matthews_corrcoef(y, top_y_predicted)
    print(svm_classreport)
    print('MCC-Score: ',svm_mcc)
   
 
#result storage
    with open(out_file_path + '50_UNTOUCHED_proteins' + str(win) + 'predictor_accuracy_results.txt', 'w') as out_file:
        out_file.write('Matthews correlation coefficient (MCC) SVM: ' + str(svm_mcc) + '\n')
        out_file.write('Classification report SVM: ' + '\n' + str(svm_classreport) + '\n')
        out_file.write('Confusion matrix SVM: ' + '\n' + str(svm_confusionm) + '\n')
    out_file.close()
#prediction result
    top_pred_list = []
    for i in range(len(input_id)):
        unknown = input_id[i]
        #single = loaded_model.predict(unknown)
        states = []
        for n, j in enumerate(top_y_predicted):
            if j == 0:
                states.append('G')
            if j == 1:
                states.append('M')
        output_pred = ''.join(states)
        top_pred_list.append(output_pred)
    
    
    output_file = open(out_file_path + 'topology_prediction_accuracy_50UNTOUCHED', 'w')
    for i in range(len(input_id)):
        output_file.write(input_id[i] + '\n')
        output_file.write(input_seq[i] + '\n')
        output_file.write(top_pred_list[i] + '\n')
    output_file.close()
    
    
    
    
    
    return svm_confusionm

    

if __name__ == '__main__':
    (np_savez('../input/membrane-alpha_2state.3line_last50_UNTOUCHED.txt',
              19,
              '../numpy/50UNTOUCHED_svm_input'))
    print(prediction_fct('../input/membrane-alpha_2state.3line_last50_UNTOUCHED.txt',
                         '../output/train_test_dataset_model_win19.sav',
                         '../numpy/50UNTOUCHED_svm_input.npz',
                         19,
                         '../output/'))
