#=========================parser=================
import numpy as np
def data_input(dataset):
    global total_array, SEQ, TOP
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
    return 'Status: parser run is complete'

##########################################################################
#=========================one hot encoding===============================#
##########################################################################    
from numpy import argmax
def input_convertion(dataset):
    # define input string
    data = data_input(dataset)
    print(data)
    #print(SEQ)
    #print(TOP)

#===============sequence===================
# definition of possible input values
    aa = 'ARNDCEQGHILKMFPSTWYV'
    
    # define a mapping of chars to integers
    aa_char_to_int = dict((c, i) for i, c in enumerate(aa))
    print(aa_char_to_int)
    
    # sequence integer encode input data
    SEQ_num=[]
    for aa_element in SEQ:
        for char in aa_element:
            aa_integer_encoded = [aa_char_to_int[char] for char in aa_element]
        SEQ_num.append(aa_integer_encoded)
    print('amino acid encoded', SEQ_num)
    
# seq one hot encode 
    aa_onehot_encoded = list()
    for element in SEQ_num:
	    for char in element:
	        letter = [0 for _ in range(len(aa))]
	        letter[char] = 1
	        aa_onehot_encoded.append(letter)
    print('one hot encoded sequence', aa_onehot_encoded)

#==================topology================    
    # definition of possible input values
    top = 'gB'
    
    # define universe of possible input values
    top_char_to_int = dict((c, i) for i, c in enumerate(top))
    print(top_char_to_int)
    
    # topology integer encode input data
    TOP_num=[]
    for top_element in TOP:
        for char in top_element:
            top_integer_encoded = [top_char_to_int[char] for char in top_element]
        TOP_num.append(top_integer_encoded)
    print('topology encoded', TOP_num)
       
# top one hot encode 
    top_onehot_encoded = list()
    for element in TOP_num:
	    for char in element:
	        letter = [0 for _ in range(len(top))]
	        letter[char] = 1
	        top_onehot_encoded.append(letter)
    print('one hot encoded topology', top_onehot_encoded)
            
#===============closing==================== 
    return 'Status: encoding run is complete'
      
if __name__ == '__main__':
    #print(data_input('../datasets/testdataset_mb2s3l.txt'))
    #print(data_input('../datasets/testdataset_aa_seq.txt'))
    #print(input_convertion('../datasets/testdataset_mb2s3l.txt'))
    print(input_convertion('../datasets/testdataset_aa_seq.txt')) 
    #print(input_convertion('../datasets/membrane-beta_2state.3line.txt')) 
       
