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
    
    #print(ID)
    #print(SEQ)
    #print(TOP)
    
#=========================sliding window=================================#
    seq = SEQ
    seq_str = ','.join(map(str, seq)).split(',')
    
    win = 3 #int(input('Please enter a window size (odd number >3):'))
    pad = win//2 #equals to two
    seq_win_list =[]
    for element in seq_str:
        for char in element:
            char = ((pad)*'0' + element + (pad)*'0')
        for i in range(1, len(char)-1):
            seq_win_list.append(char[i-pad:i+pad+1])
        
    a = 'Status: parser + sliding window (windowsize:'    
    b = ') run is complete'
#=================closing======================
    return  a, win, b #'Status: parser + sliding window run is complete' # 


##########################################################################
#=========================one hot encoding===============================#
##########################################################################    
from numpy import argmax
def input_convertion(dataset):
    global aa_onehot_encoded, TOP_num
    # define input string
    data = data_input(dataset)
    print(data)
    seq_data = seq_win_list
    top_data = TOP
#===============sequence===================
# definition of possible input values
    aa = '0ARNDCEQGHILKMFPSTWYV'
    
    # define a mapping of chars to integers
    aa_char_to_int = dict((c, i) for i, c in enumerate(aa))
    
    # sequence integer encode input data
    SEQ_num=[]
    for aa_element in seq_data:
        #print(aa_element)
        temp_list=[]
        for char in aa_element:
            aa_integer_encoded = [aa_char_to_int[char] for char in aa_element] #temp_list.extend([aa_char_to_int[char] for char in aa_element]
        SEQ_num.append(aa_integer_encoded) #temp_list
    #print('amino acid encoded', SEQ_num)


    # seq one hot encode 
    aa_onehot_encoded = list()
    for element in SEQ_num:
        temp_list = []
        for char in element:
            letter = [0 for _ in range(len(aa) - 1)]
            if char != 0: 
                letter[char-1] = 1
            else:
                pass
            temp_list.extend(letter)
        aa_onehot_encoded.append(temp_list)
    #print('one hot encoded sequence', aa_onehot_encoded)
    print('aa_array size: ', np.array(aa_onehot_encoded).shape,', aa map len:', len(aa))

#==================topology=================================    
    # definition of possible input values
    top = 'gB'
    
    # define universe of possible input values
    top_char_to_int =  dict((c, i) for i, c in enumerate(top)) #{'g': 0, 'B': 1}
    #top_decode = int_to_char = dict((i, c) for i, c in enumerate(top))
    #print(top_char_to_int)
    
    # topology integer encode input data
    TOP_num=[]
    for top_element in top_data:
        for char in top_element:
            top_integer_encoded = [top_char_to_int[char] for char in top_element]
        TOP_num.extend(top_integer_encoded) # extend to append?
    #print('XXXXXXX', TOP_num)
    
#==========saving the one hot vectors as numpy files=========
    print(len(aa_onehot_encoded))
    print(len(TOP_num))
    np.savez_compressed('../numpy/svm_input',
                       aa_onehot_encoded=aa_onehot_encoded,
                       TOP_num=TOP_num)

#===============closing==============================
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
                    
                    Have a good day!'''

###############################################################################################
#===============================SVM training from Linnea======================================#
###############################################################################################
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

#takes a np.savez file and uses it as input. Will add option to change the parameters of the SVM, generate several models, generate output model file names that correspond to the parameters and model type.  
def svm_fct(svm_input,svm_output): 
    loaded_model = joblib.load(svm_output)
    predicted = loaded_model(x)
    print(predicted)
    
    clf = joblib.load(svm_input)
    predicted=clf.predict(map_word)
    print(predicted)
    print("This predictor has a cross-validation accuracy of 0.97")

    #Put the output back to the features, S and G

    structure_dict = { 1:'G', 2:'S'}

    m=predicted.tolist()

    struct_prediction=[]
    for i in m:
	    e = structure_dict[i]
	    struct_prediction.append(e)

    print (struct_prediction)

    #Save the prediction output in a file 

    with open ('//home/u2195/Desktop/Dropbox/Bioinformatics_projects/results/' + 'SP_Prediction' '.fasta', 'w')as b:
	    for i in range(len(titlelist)):
		    b.write('Prediction of Signal Peptide by Carolina Savatier'+'\n')
		    b.write(titlelist[i]+'\n')
		    b.write(seqlist[i]+'\n')
    b.write(''.join(struct_prediction)+'\n')
        
        
    
    
           
if __name__ == '__main__':
    #print(input_convertion('../datasets/testdataset_mb2s3l.txt'))
    print(input_convertion('../datasets/membrane-beta_2state.3line.txt')) 
    print(svm_fct('../numpy/svm_input.npz','../output/model.pklv'))
