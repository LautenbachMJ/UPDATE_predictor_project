'''Maximilian Julius Lautenbach'''
##############################################################################################
#=============================== Prediction =================================================#
##############################################################################################
from def_predictor_readin import prediction
def combine(svm_output, seq_to_pred, pred_output_path):
    '''prediction of topology based on an iput sequence and a trained model'''
    prediction(svm_output, seq_to_pred, pred_output_path)
    output_file = open(pred_output_path + seq_to_pred[9:] + '_predicted.txt', 'r')
    output_f = output_file.readlines()[21:]
    for element in output_f:
        print(element.rstrip())
    output_file.close()
    return '\n' + 'The result file is saved here:' + pred_output_path + seq_to_pred[9:]+ '_predicted.txt' + '\n'
if __name__ == '__main__':
   print(combine('../output/train_test_dataset_model_win19.sav',
                 '../input/test_pred_3.txt', #change here
                 '../output/'))
