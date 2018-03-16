'''Maximilian Julius Lautenbach'''
import pickle
import numpy as np
###############################################################################################
#=============================== Prediction =================================================#
###############################################################################################
def data_input(seq_to_pred):
    '''read in of data file'''
    filehandle = open(seq_to_pred, 'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    input_id = lines[0::2]
    input_seq = lines[1::2]
    return input_id, input_seq
#========================= sliding window =================================
def window(seq_to_pred):
    '''sliding window'''
    input_seq = data_input(seq_to_pred)[1]
    win = 19 #seems to be a good window size
    pad = win//2 #equals to two
    seq_win_list = []
    temp = []
    for element in input_seq:
        char = ((pad)*'0' + element + (pad)*'0')
        temp.append(char)
    for padded_seq in temp:
        tempo = []
        for residue  in range(len(padded_seq)):
            #print(residue)
            if residue+win > len(padded_seq):
                break
            tempo.append(padded_seq[residue:residue+win])
        seq_win_list.append(tempo)
    return seq_win_list, win
#========================== aa to num =====================================
def a_to_num(seq_to_pred):
    '''aa to num'''
    seq_data = window(seq_to_pred)[0]
    aa_map = {'K': 12, 'Q': 7, 'I': 10, 'W': 18, 'T': 17,
              'R': 2, 'V': 20, 'E': 6, '0': 0, 'H': 9, 'P': 15, 'G': 8,
              'N': 3, 'A': 1, 'Y': 19, 'L': 11, 'F': 14, 'D': 4, 'M': 13, 'S': 16, 'C': 5}
    seq_num = []
    for sequence in seq_data:
        temp = []
        for windows in sequence:
            encoded = [aa_map[aa] for aa in windows]
            temp.append(encoded)
        seq_num.append(temp)
    return seq_num
#========================== num to binary ==================================
def one_hot(seq_to_pred):
    '''num to binary'''
    one_hot_input = a_to_num(seq_to_pred)
    aa_list = '0ARNDCEQGHILKMFPSTWYV'
    aa_onehot_encoded = []
    for protein in one_hot_input:
        this_prot = []
        for win in protein:
            this_win = []
            for residue in win:
                letter = [0 for _ in range(len(aa_list))]
                if residue != 0:
                    letter[residue] = 1
                this_win.extend(letter)
            this_prot.append(this_win)
        aa_onehot_encoded.append(this_prot)
    aa_onehot_encoded = np.array(aa_onehot_encoded)
    return aa_onehot_encoded
############################################################################
#======================= Topology prediction ==============================#
############################################################################
def prediction(svm_output, seq_to_pred, pred_output_path):
    '''Topology prediction'''
    loaded_model = pickle.load(open(svm_output, 'rb'))
    pred_input_encoded = one_hot(seq_to_pred)
    input_id = data_input(seq_to_pred)[0]
    input_seq = data_input(seq_to_pred)[1]
    top_pred_list = []
    for i in range(len(pred_input_encoded)):
        unknown = pred_input_encoded[i]
        single = loaded_model.predict(unknown)
        states = []
        for n, j in enumerate(single):
            if j == 0:
                states.append('G')
            if j == 1:
                states.append('M')
        output_pred = ''.join(states)
        top_pred_list.append(output_pred)
    finish = ('''Status: Prediction run is complete!
                   
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
                    
       I did the work for you and predicted all the topologies for you.
                   
            Have a good day and now you can monkey around!
            ''')
    output_file = open(pred_output_path + seq_to_pred[9:] + '_predicted.txt', 'w')
    output_file.write(finish + '\n' + 'Please find the prediction results below:' + '\n' + '\n')
    for i in range(len(input_id)):
        output_file.write(input_id[i] + '\n')
        output_file.write(input_seq[i] + '\n')
        output_file.write(top_pred_list[i] + '\n')
    output_file.close()
    return finish
#if __name__ == '__main__':
#    (prediction(svm_output, seq_to_pred, pred_output_path))
