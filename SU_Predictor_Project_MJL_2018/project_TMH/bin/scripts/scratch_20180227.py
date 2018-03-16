#=========================parser=================
import numpy as np
def data_input(dataset):
    global total_array, SEQ, TOP, seq_win_list
    #opening all the necessary files,dicts and lists
    filehandle = open(dataset,'r')
    #filehandle.close()
    dictionary = {}
    ID = []
    SEQ = []
    TOP = []
    #stip the lines in the file and sort the lines into lists
    lines = [line.strip() for line in filehandle]
    ID = lines[0::3]
    SEQ = lines[1::3]
    TOP = lines[2::3]
    
    print(ID)
    print(SEQ)
    print(TOP)
            
#=================closing======================
    #return 'Status: parser + sliding window run is complete'
    
##########################################################################
#=========================sliding window=================================#
##########################################################################

    seq = SEQ
    seq_str = ','.join(map(str, seq)).split(',')
    
    win = 3
    pad = win//2 #equals to two
    seq_win_list =[]
    for element in seq_str:
        #print(element)
        for char in element:
            char = ((pad)*'0' + element + (pad)*'0')
        #print('starting sequence:', char)
        #print('window size:', win)
        for i in range(1, len(char)-1):
                #print(i)
            seq_win_list.append(char[i-pad:i+pad+1])
        
        

#=================closing======================
    return 'Status: parser + sliding window run is complete'


##########################################################################
#=========================one hot encoding===============================#
##########################################################################    
from numpy import argmax
def input_convertion(dataset):
    # define input string
    data = data_input(dataset)
    print(data)
    #print(seq_win_list)
    seq_data = seq_win_list
    #print(seq_data)
    top_data = TOP
#===============sequence===================
# definition of possible input values
    aa = '0ARNDCEQGHILKMFPSTWYV'
    
    # define a mapping of chars to integers
    aa_char_to_int = dict((c, i) for i, c in enumerate(aa))
    #print(aa_char_to_int)
    
    # sequence integer encode input data
    SEQ_num=[]
    for aa_element in seq_data:
        #print(aa_element)
        temp_list=[]
        for char in aa_element:
        
            #temp_list=[]
            aa_integer_encoded = [aa_char_to_int[char] for char in aa_element]
            #temp_list.extend(aa_integergedit_encoded)
            #print(aa_integer_encoded)
        SEQ_num.append(aa_integer_encoded)
    print('amino acid encoded', SEQ_num)


    # seq one hot encode 
    aa_onehot_encoded = list()
    for element in SEQ_num:
        temp_list = []
        for char in element:
            #if char ==  '0':
            letter = [0 for _ in range(len(aa) - 1)]
            if char != 0: 
                letter[char-1] = 1
            else:
                pass
            temp_list.extend(letter)
        aa_onehot_encoded.append(temp_list)
    print('one hot encoded sequence', aa_onehot_encoded)
    print(np.array(aa_onehot_encoded).shape, len(aa))

#==================topology================    
    # definition of possible input values
    top = 'gB'
    
    # define universe of possible input values
    top_char_to_int = dict((c, i) for i, c in enumerate(top))
    print(top_char_to_int)
    
    # topology integer encode input data
    TOP_num=[]
    for top_element in top_data:
        for char in top_element:
            top_integer_encoded = [top_char_to_int[char] for char in top_element]
        TOP_num.append(top_integer_encoded)
    #print('topology encoded', TOP_num)
   
       
# top one hot encode 
    top_onehot_encoded = list()
    for element in TOP_num:
	    for char in element:
	        letter = [0 for _ in range(len(top))]
	        letter[char] = 1
	        top_onehot_encoded.append(letter)
    #print('one hot encoded topology', top_onehot_encoded)
    
#==========saving the one hot vectors as numpy files=========
    print(len(aa_onehot_encoded))
    print(len(top_onehot_encoded))
    np.savez_compressed('../numpy/svm_input',
                       aa_onehot_encoded=aa_onehot_encoded,
                       top_onehot_encoded=top_onehot_encoded)
    #np.save('../numpy/svm_aa_input.npy',aa_onehot_encoded)
    #np.save('../numpy/svm_top_input.npy',top_onehot_encoded)
#===============closing==============================
    return 'Status: encoding run is complete'
    
  
  
if __name__ == '__main__':
    #print(data_input('../datasets/testdataset_mb2s3l.txt'))
    #print(data_input('../datasets/testdataset_aa_seq.txt'))
    #print(data_input('../datasets/testdataset_aa_seq_one.txt'))
    #print(sliding_window()
    #print(input_convertion('../datasets/testdataset_mb2s3l.txt'))
    #print(input_convertion('../datasets/testdataset_aa_seq.txt'))
    #print(input_convertion('../datasets/testdataset_aa_seq_one.txt')) 
    print(input_convertion('../datasets/membrane-beta_2state.3line.txt')) 
