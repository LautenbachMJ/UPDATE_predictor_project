'''Maximilian Julius Lautenbach'''
def data_input(dataset):
    filehandle = open(dataset,'r')
    lines = [line.strip() for line in filehandle]
    filehandle.close()
    input_id = lines[0::3]
    input_seq = lines[1::3]
    input_top = lines[2::3]
    for i in range(len(input_id)):
        s=open('../datasets/PSSM_files/single_fastas/'+input_id[i]+'.fasta','w')
        s.write(input_id[i]+'\n')
        s.write(input_seq[i]+'\n')
        s.close()
    return 'done' 
if __name__ == '__main__':
    print(data_input('../datasets/membrane-alpha_2state.3line_train_test.txt'))
