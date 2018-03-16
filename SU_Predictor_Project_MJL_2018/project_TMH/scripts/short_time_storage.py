with open(out_file_path + '50_UNTOUCHED_proteins' + str(win) + 'predictor_accuracy_results.txt', 'w') as out_file:
        out_file.write('Cross-validation scores for SVC: ' + str(cvs_svm_mean)+ '\n')
        out_file.write('Matthews correlation coefficient (MCC) SVM: ' + str(svm_mcc) + '\n')
        out_file.write('Classification report SVM: ' + '\n' + str(svm_classreport) + '\n')
        out_file.write('Confusion matrix SVM: ' + '\n' + str(svm_confusionm) + '\n')
    out_file.close()
