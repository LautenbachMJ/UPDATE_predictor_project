# New GitHub Repository for my predictor project "Alpha helical TM proteins (2 state)"


**The final predictor** file (*predictor_only.py*) can be found when following this path: UPDATE_predictor_project/SU_Predictor_Project_MJL_2018/project_TMH/scripts/

This predictor uses a model (*../output/train_test_dataset_model_win19.sav*) that was generated 
(*svm_model_creator.py*) and trained with the dataset "train_test" (263 proteins) using a 
window size of 19 and following SVM parameter (gamma=0.1, kernel='linear', C=1.0).
The final predictor parses and encodes the input file by using a background file (*def_predictor_readin.py*) and outputs the predicted results in the terminal and saves it in an text file (*../output/XX_predicted.txt*).
For prediction I would suggest two files, that are located in the input directery (../input/test_pred_3.txt or PDB_pred.txt).
Just copy/paste the path+file name  the to the provided spot in line 17 (comment with "change here") of the predictor file (predictor_only.py).

**Predictor accuracy test on 50 proteins** was performed on the file 50UNTOUCHED (../input/membrane-alpha_2state.3line_last50_UNTOUCHED50.txt and were parsed, encoded, predicted and scored with a separate file (predictor_accuracy_50_UNTOUCHED_proteins.py) based on the final predictor model.
Both the prediction results and the accurary scoring/calculation were saved as seperate files in the output directory (../output/topology_prediction_accuracy_50UNTOUCHED.txt or 50_UNTOUCHED_proteins19predictor_accuracy_results.txt) 



**The PSSM based predictor** is unfortunately unfinished/not working, because I realised that I was working with the wrong dataset until a few days ago. With the wrong dataset, the PSSM predictor was working. Anyway, the PSSM based model creator does not work because of an error in the parsing step of the PSSMs.
The PSSMs can found by following this path: 
UPDATE_predictor_project/SU_Predictor_Project_MJL_2018/project_TMH/datasets/PSSM_files/PSSMs/
All files that belong to the PSSM based predictor can be found in the scripts directory (pssm_parsing.py, pssm_predicting.py






