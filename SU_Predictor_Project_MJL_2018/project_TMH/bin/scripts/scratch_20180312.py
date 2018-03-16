def data_input(dataset):
    g = open(dataset,'r')
    g = g.read().splitlines()
    datalist = list()
    for line in g:
	    newline = line.strip()
	    datalist.append (newline)
    titlelist = list ()
    for i in datalist [:: 3]:
	    titlelist.append (i)



'''
    #Define window size
    window = 35
    pad_size = 17

    bigwordlist=[]
    structvectorlist=[]

    #Open all the fastafiles that are in the titlelist
    for r in titlelist:
	    line_list =[]
	    for line in g[3:-7]:
		    newline = line.split()
		    newline = newline [22:42]
		    line_list.append (newline)

		
    #Normalize the values because they are in percentage

	    for i in line_list: 
		    for j in range (0, len (i)):
			    i[j] = int(i[j])/100
	
    #Padding, now we have vectors directly, so the padding is done by adding vectors containing 20 zeros. 
	    temp_prot=[]
	    a=list(np.zeros(20))
	    for i in range (0, pad_size):
		    temp_prot.append(a)
	    temp_prot.extend(line_list)
	    for i in range (0, pad_size):
		    temp_prot.append(a)
		
	    #print(temp_prot)
	    #print(len(temp_prot))
	
    #Create words with pssm information
	    wordlist=[]
	    for i in range (0, len (temp_prot)-(window-1)):
		    b=temp_prot[i:i+(window)]
		    b = [j for i in b for j in i]
		    if len(b) != window*20:
			    print ("oh no")
		    wordlist.append(b)
	    #print(wordlist)
	    #print(len(wordlist))
	    bigwordlist.append(wordlist)

    bigwordlist=[j for i in bigwordlist for j in i]

    #print (bigwordlist)
'''
if __name__ == '__main__':
    print(data_input('../datasets/PSSM_files/PSSMs/*.pssm'))
