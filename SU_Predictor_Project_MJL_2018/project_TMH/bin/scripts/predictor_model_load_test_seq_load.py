##############################################################################################
#=============================== Prediction =================================================#
##############################################################################################
from sklearn import svm
import pickle
import numpy as np
def data_input(dataset):
    filehandle = open(dataset,'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    
    ID = []
    SEQ = []
    
    ID = lines[0::2]
    SEQ = lines[1::2]
#=========================sliding window=================================#
    seq = SEQ
    seq_str = ','.join(map(str, seq)).split(',')
    
    win = 31 #seems to be a good window size
    pad = win//2 #equals to two
    seq_win_list =[]
    for element in seq_str:
        char = ((pad)*'0' + element + (pad)*'0')
        for i in range(pad, len(char)-pad):
            seq_win_list.append(char[i-pad:i+pad+1])
        
    return ID, SEQ, seq_win_list, win 

def seq_input_convertion(dataset): #,conv_output):
    data = data_input(dataset)
    #data structure: SEQ[0], TOP[1], seq_win_list[2]
    seq_data = data[2] #data_input(seq_win_list)
    print('Sequence: loaded!')
    return seq_data
   
def one_hot_seq(dataset):    
#===============sequence===================
    seq_data = seq_input_convertion(dataset)
    #print(seq_data)
    
    # definition of possible input values & define a mapping of chars to integers
    aa = '0ARNDCEQGHILKMFPSTWYV'
    aa_char_to_int = {'K': 12, 'Q': 7, 'I': 10, 'W': 18, 'T': 17,
     'R': 2, 'V': 20, 'E': 6, '0': 0, 'H': 9, 'P': 15,   'G': 8,
      'N': 3, 'A': 1, 'Y': 19, 'L': 11, 'F': 14, 'D': 4, 'M': 13, 'S': 16, 'C': 5}       
    
    # sequence integer encode input data
    SEQ_num=[]
    for aa_element in seq_data:
        temp_list=[]
        for residue in aa_element:
            aa_integer_encoded = [aa_char_to_int[residue] for residue in aa_element]
        SEQ_num.append(aa_integer_encoded)
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
    #print(aa_onehot_encoded)    
    print('Sequence numbers into binary convertion: done!')
    return aa_onehot_encoded


def prediction(svm_output, seq_to_pred):    
    # load the model from disk
    loaded_model = pickle.load(open(svm_output, 'rb'))
    
    pred_input = one_hot_seq(seq_to_pred)
       
    print('''==================== Prediction ==========================
    
    ''')
    
    #print('Parameter of the model used for topology prediction:', loaded_model)
    #print('Input sequence length:',len(pred_input))
    
    print(pred_input)
    '''
    result = loaded_model.predict(pred_input) 
    results = list(result)
    
    for n, i in enumerate(results):
        if i == 0:
            results[n] = 'g'
        if i == 1:
            results[n] = 'B'
    
    output_id = open(seq_to_pred,'r').readlines()[0].replace('\n','')
    output_seq = open(seq_to_pred,'r').readlines()[1].replace('\n','')    
    
    print('Predicted topology: ')
    output_pred = ''.join(results)
    
    print(output_id)
    print(output_seq)
    print(output_pred)    
    
    return  '\n Have a good day, now you can monkey around!'
'''

    
if __name__ == '__main__':
    #print(one_hot_seq('../datasets/test_pred.txt'))
    print(prediction('../output/model.sav', '../datasets/test_pred_3.txt'))
    

