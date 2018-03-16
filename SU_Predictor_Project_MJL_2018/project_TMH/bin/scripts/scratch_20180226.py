def sliding_window(sequence, window_size):
    seq = sequence
    win = window_size
    #print(seq, win)
    pad = win//2 #equals to two
    for char in seq:
        char = ((pad)*'0' + seq + (pad)*'0')
    print('starting sequence:', char)
    print('window size:', win)
    
    win_list =[]    
    for i in range(1, len(char)-1):
            print(i)
            win_list.append(char[i-pad:i+pad+1])
            print(win_list)
    return 'done'

    	                                                                        
    #for e in it: # Subsequent windows
    #   win[:-1] = win[1:]
      #  win[-1] = e
      #  yield win

if __name__=="__main__":
    print(sliding_window('QAM',3))
    
  






#seq = "QAMLCSRT"
#win = 3
#pad = win//2 
#for i in range(1, len(seq)-1):
#	print(i)
#	print(seq[i-pad:i+pad+1])
#
#window = 7
#padding = window//2 #equals to two
#
#for char in seq:
#    char = ((padding)*'0' + seq + (padding)*'0')
#print(char)
#for residue in range(0, len(char)):
#    if residue+(window) > len(char):
#    #break
#        print('residue', residue)
#for i in range(1, len(char)-1):
#	print(i)
#print(char[i-pad:i+pad+1])
