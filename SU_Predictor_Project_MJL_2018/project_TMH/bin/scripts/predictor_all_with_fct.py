#=========================parser=================
import numpy as np
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
    
#=========================sliding window=================================#
    seq = SEQ
    seq_str = ','.join(map(str, seq)).split(',')
    
    win = 21 #int(input('Please enter a window size (odd number >3):'))
    pad = win//2 #equals to two
    seq_win_list =[]
    for element in seq_str:
        char = ((pad)*'0' + element + (pad)*'0')
        for i in range(pad, len(char)-pad):
            seq_win_list.append(char[i-pad:i+pad+1])
    #print('reading file in and creating window: done!')        
    
    return SEQ, TOP, seq_win_list 


##########################################################################
#=========================one hot encoding===============================#
##########################################################################    

def seq_input_convertion(data): #,conv_output):
    # define input string
    data = data_input(dataset)
    #data structure: SEQ[0], TOP[1], seq_win_list[2] 
    seq_data = data[2] #data_input(seq_win_list)
    return seq_data

def top_input_convertion(data): #,conv_output):
    # define input string
    data = data_input(dataset)
    #data structure: SEQ[0], TOP[1], seq_win_list[2] 
    top_data = data[1] #TOP
    print(top_data)
    return top_data
    
def one_hot_seq(one_hot_input):    
#===============sequence===================
    #import input from function above
    one_hot_input = seq_input_convertion(dataset)
    seq_data = one_hot_input(datasetone_hot_input)
    print('is this seq?',seq_data)
    
    
    # definition of possible input values & define a mapping of chars to integers
    aa = '0ARNDCEQGHILKMFPSTWYV'
    aa_char_to_int = dict((c, i) for i, c in enumerate(aa))
    
           
    # sequence integer encode input data
    SEQ_num=[]
    for aa_element in seq_data:
        #print(aa_element)
        temp_list=[]
        for residue in aa_element:
            #temp_list=[]
            aa_integer_encoded = [aa_char_to_int[residue] for residue in aa_element]
            #temp_list.extend(aa_integergedit_encoded)
        SEQ_num.append(aa_integer_encoded)
    #print('amino acid encoded', len(SEQ_num))

    # seq one hot encode 
    aa_onehot_encoded = list()
    for element in SEQ_num:
        temp_list = []
        for char in element:
            letter = [0 for _ in range(len(aa))]
            if char != 0: 
                letter[char] = 1
            else:
                pass
            temp_list.extend(letter)
        aa_onehot_encoded.append(temp_list)
    print('hallo')
    return aa_onehot_encoded
#///////////////////////////////////////////////////////////////   
    
    #print('one hot encoded sequence', aa_onehot_encoded)
    #print('aa_array size: ', np.array(aa_onehot_encoded).shape,', aa map len:', len(aa))

def top_encoding(top_encode_input): 
#==================topology=================================    
    #import input from function above
    top_encode_input = top_input_convertion(dataset)
    top_data = top_encode_input
    
    
    # definition of possible input values &define universe of possible input values
    top = 'gB'
    top_char_to_int =  dict((c, i) for i, c in enumerate(top))
        
    
    # topology integer encode input data
    TOP_num=[]
    for proteins in top_data:    
        for tops in proteins:
            t = top_char_to_int[tops]
            TOP_num.append(t)
    return TOP_num
#///////////////////////////////////////////////////////////////       
    
    #print(len(aa_onehot_encoded))
    #print(len(TOP_num))
    
#==========saving the one hot vectors as numpy files=========
def np_savez(conv_output):
    aa_onehot_encoded = one_hot_seq(one_hot_input)
    TOP_num = top_encoding(top_encode_input)
    
    np.savez_compressed(conv_output,
                       aa_onehot_encoded=aa_onehot_encoded,
                       TOP_num=TOP_num)

#===============closing=========================================
    return '''Status: encoding run is complete!
                   
                        .="=.
                      _/.-.-.\_     _
                     ( ( o o ) )    ))
                      |/  "  \|    //
      .-------.        \"---"/    //
     _|~~ ~~  |_       /`"""`\\  ((
   =(_|_______|_)=    / /_,_\ \\  \\
     |:::::::::|      \_\\_"__/ \  ))
     |:::::::[]|       /`  /`~\  |//
     |o=======.|      /   /    \  /
     `"""""""""`  ,--`,--"\/\    /
                   "-- "--"  "--" 
                    
                   I did the work for you and saved a numpz file for you.
                   
                    Have a good day!'''
    

###############################################################################################
#==========================================SVM================================================#
###############################################################################################
from sklearn import svm
import pickle

def svm_fct(svm_input,svm_output):
    loaded = np.load(svm_input)
    np.set_printoptions(threshold=np.inf)
    
    aa_list = loaded['aa_onehot_encoded']
    top_list = loaded['TOP_num']
   
    ###################################   TRAINING & Saving   ################################
    x_train, y_train = aa_list, top_list
    x,y = x_train, y_train
    #print(x_train)
    
    clf = svm.SVC()
    clf.fit(x, y)
    print(clf)

    pickle.dump(clf, open(svm_output, 'wb'))
    print(x_train)    

##############################################################################################
#=============================== Prediction =================================================#
##############################################################################################
'''    
def prediction(svm_output, seq_to_pred):    
    # load the model from disk
    loaded_model = pickle.load(open(svm_output, 'rb'))
    loaded_seq = open(seq_to_pred, 'r')
    seq_pred = [line.strip() for line in loaded_seq]
    print(seq_pred)
    
    ############## input encoding into binary ################
    
    input_seq_num=[]
    for aa_element in seq_pred:
        for residue in aa_element:
            input_seq_num = [aa_char_to_int[residue] for residue in aa_element]
    #print(input_seq_num)
        
    #print(len(input_seq_num))

    # seq one hot encode 
    max_len = 21
    seq_oh_pred = list()
    temp_list = []
    for element in input_seq_num:
        #print(element)
        letter = [0 for _ in range(max_len)]
        if element != 0: 
            letter[element] = 1
        else:
            pass
        #temp_list.extend(letter)
        seq_oh_pred.append(letter)
    print(seq_oh_pred)
    
    
      
    result = loaded_model.predict(seq_oh_pred) 
    print(result)
    results =list(result)
    
    for n, i in enumerate(results):
        if i == 0:
            results[n] = 'g'
        if i == 1:
            results[n] = 'B'
    return results
'''    

    
if __name__ == '__main__':
    #print(data_input('../datasets/testdataset_aa_seq_B.txt'))
    #print(one_hot_seq('../datasets/testdataset_aa_seq_B.txt'))
    print(one_hot_seq('../datasets/membrane-beta_2state.3line_10.txt'))
    #print(np_savez('../numpy/svm_input'))
    #print(input_convertion('../datasets/testdataset_aa_seq_B.txt','../numpy/svm_input'))
    #print(svm_fct('../numpy/svm_input.npz','../output/model.sav'))
        #print(prediction('../output/model.sav','../datasets/test_pred.txt'))
    #print(input_convertion('../datasets/membrane-beta_2state.3line_10.txt','../numpy/svm_input')) 
    #print(svm_fct('../numpy/svm_input.npz','../output/model.sav'))
    
    #print(input_convertion('../datasets/testdataset_mb2s3l.txt'))
    #print(input_convertion('../datasets/testdataset_aa_seq_one.txt'))
    #print(input_convertion('../datasets/testdataset_mb2s3l.txt'))
    #print(input_convertion('../datasets/membrane-beta_2state.3line_10.txt')) 
    #print(svm_fct('../numpy/svm_input.npz','../output/model.sav'))
