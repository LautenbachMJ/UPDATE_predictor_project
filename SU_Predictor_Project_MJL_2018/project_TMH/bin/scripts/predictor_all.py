############################################################################
#=============================== parser ===================================#
############################################################################
import numpy as np
from numpy import array

def data_input(dataset):
    #opening all the necessary files,dicts and lists
    filehandle = open(dataset,'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    
    ID = []
    SEQ = []
    TOP = []
    #stip the lines in the file and sort the lines into lists
    ID = lines[0::3]
    SEQ = lines[1::3]
    TOP = lines[2::3]
    return SEQ, TOP 
       
#======================== sliding window ===================================
def window(dataset):
    seq = data_input(dataset)[0]
    
    win = 19 #literature window size (TMSEG) but 31 seems to be a good window size too
    pad = win//2 #equals to two
    
    seq_win_list =[]
    temp=[]
    for element in seq:
        char = ((pad)*'0' + element + (pad)*'0')
        temp.append(char)
        #print(temp)
        #for i in range(pad, len(char)-pad):
    for padded_seq in temp:
        tempo = []
        for residue  in range(len(padded_seq)):
            #print(residue)
            if residue+win > len(padded_seq):
                break
            tempo.append(padded_seq[residue:residue+win])
        seq_win_list.append(tempo)

    return seq_win_list, win 


#========================= aa into num ======================================
def a_to_num(dataset):    
    seq_data = window(dataset)[0]
    
    aa = '0ARNDCEQGHILKMFPSTWYV'
    aa_map = {'K': 12, 'Q': 7, 'I': 10, 'W': 18, 'T': 17,
     'R': 2, 'V': 20, 'E': 6, '0': 0, 'H': 9, 'P': 15,   'G': 8,
      'N': 3, 'A': 1, 'Y': 19, 'L': 11, 'F': 14, 'D': 4, 'M': 13, 'S': 16, 'C': 5}       
    
    SEQ_num=[]
    for sequence in seq_data:
        temp=[]
        for windows in sequence:
            encoded= [aa_map[aa] for aa in windows]
            temp.append(encoded)
        SEQ_num.append(temp)
    return SEQ_num

#======================== num into binary ==================================    
def one_hot(dataset):
    one_hot_input = a_to_num(dataset)
    aa = '0ARNDCEQGHILKMFPSTWYV'
       
    aa_onehot_encoded =[]
    for protein in one_hot_input:
        this_prot=[]
        for win in protein:
            this_win=[]
            for residue in win:
                letter = [0 for _ in range(len(aa))]
                if residue != 0: 
                    letter[residue] = 1
                this_win.extend(letter)
            this_prot.append(this_win)
        aa_onehot_encoded.extend(this_prot)
    #print(aa_onehot_encoded)
    aa_onehot_encoded = np.array(aa_onehot_encoded)
    #print(aa_onehot_encoded.shape)
    return aa_onehot_encoded

#================== topology into num ======================================    
def top_encoding(dataset): 
    top_data = data_input(dataset)[1]
       
    top = 'gB'
    top_char_to_int =  dict((c, i) for i, c in enumerate(top))
        
    # topology integer encode input data
    TOP_num=[]
    for proteins in top_data:
        temp=[]    
        for tops in proteins:
            t = top_char_to_int[tops]
            temp.append(t)            
        TOP_num.extend(temp)
    print('Topology into numbers convertion: done')
    return TOP_num
    
#========== saving encoded sequences & topologies into np.file =============
def np_savez(dataset,conv_output):
    aa_list = one_hot(dataset)
    top_list = top_encoding(dataset)

    np.savez(conv_output, x=aa_list, y=top_list)
    return aa_list, top_list
    
############################################################################
#=========================== SVM Training =================================#
############################################################################
from sklearn import svm
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def svm_fct(infile,outfile): 
    np.set_printoptions(threshold=np.inf)    
    
    loaded = np.load(infile)
    X = loaded['x']
    Y = loaded['y']
    print (X.shape, Y.shape)
    #print(X)
    #print(Y)
    
    clf = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(X,Y) 
    print(clf)
    
    pickle.dump(clf, open(outfile, 'wb'))
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


  
    
#    ###################################   TRAINING & Saving   ################################
#    #X,y = aa_list, top_list
#    
#    #clf = svm.SVC()
#    #clf.fit(X, y)
#    #print(clf)
#    
#    ################################# Cross-validation from Lucie########################################
#    
#    #cross_val = 5 #int(input('Fold of cross-validation: '))
#    
#    #clf = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0, cache_size=200).fit(x,y) 
#    #cvs = cross_val_score(clf, aa_list, top_list, cv = cross_val,n_jobs=-1) #n_jobs=-1 makes it faster
#    #avg = np.average(cvs) 
#    #print('Cross-validation accuracy with', cross_val, 'x fold:',avg)
    
    
if __name__ == '__main__':
    #print(np_savez('../datasets/membrane-beta_2state.3line_10.txt','../numpy/svm_input'))
    print(np_savez('../datasets/membrane-beta_2state.3line_10.txt','../numpy/svm_input'))
    print(svm_fct('../numpy/svm_input.npz','../output/model.sav'))
    #print(svm_fct('../numpy/svm_input.npz','../output/model.sav'))
    #print(top_encoding('../datasets/testdataset_aa_seq.txt'))
    #print(one_hot('../datasets/testdataset_aa_seq.txt'))
    
