'''Maximilian Julius Lautenbach'''
import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
############################################################################
#=============================== parser ===================================#
############################################################################
def data_input(dataset):
    '''pareses input data'''
    filehandle = open(dataset, 'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
#   prot_id = lines[0::3]
    prot_seq = lines[1::3]
    prot_top = lines[2::3]
    return prot_seq, prot_top
# ======================== sliding window ==================================
def window(dataset):
    '''creates windows in sequence'''
    seq = data_input(dataset)[0]
    win = 19 #literature window size (TMSEG) but 31 seems to be a good window size too
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
def a_to_num(dataset):
    '''transforms aa into numbers'''
    seq_data = window(dataset)[0]
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
def one_hot(dataset):
    '''encodes sequence numbers into binary'''
    one_hot_input = a_to_num(dataset)
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
def top_encoding(dataset):
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
def np_savez(dataset, conv_output):
    '''saves the model as a np file'''
    aa_list = one_hot(dataset)
    top_list = top_encoding(dataset)
    np.savez(conv_output, x=aa_list, y=top_list)
    return aa_list, top_list
############################################################################
#=========================== SVM Training =================================#
############################################################################
def svm_fct(infile, outfile):
    '''svm code'''
    np.set_printoptions(threshold=np.inf)
    loaded = np.load(infile)
    seq = loaded['x']
    top = loaded['y']
    clf = svm.SVC(gamma=0.1, kernel='linear', C=1.0).fit(seq, top)
#    print(clf)
    pickle.dump(clf, open(outfile, 'wb'))
    cvs = cross_val_score(clf, seq, top, cv=5, n_jobs=-1)
    avg = np.average(cvs)
    print('Cross-validation accuracy with 5 x fold:', avg)
    return '''Status: Creating a prediction model: complete!
                   
                        .="=.
                      _/.-.-.\_     _
                     ( ( o o ) )    ))
                      |/  "  \|    //
      .-------.        \"---"/    //
     _|~~ ~~  |_       /`MJL`\\  ((
   =(_|_______|_)=    / /_,_\ \\  \\
     |:::::::::|      \_\\_"__/ \  ))
     |:::::::[]|       /`  /`~\  |//
     |o=======.|      /   /    \  /
     `"""""""""`  ,--`,--"\/\    /
                   "-- "--"  "--" 
                    
       I did the work for you and trained a model for you.
                   
            Have a good day and don't monkey around!'''
if __name__ == '__main__':
    (np_savez('../datasets/membrane-alpha_2state.3line_train_test.txt', '../numpy/svm_input_train_test'))
    print(svm_fct('../numpy/svm_input_train_test.npz', '../output/train_test_dataset_model_win19.sav'))
