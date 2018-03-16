# Project Alpha helical TM proteins (2 state) predictor
how to write in that diary: https://help.github.com/articles/basic-writing-and-formatting-syntax/

The prediction model was build on a small dataset (10 protein sequences; "/datasets/membrane-beta_2state.3line_10.txt").
Window size for the model is 19 residues and the svm settings are the following: gamma=0.001, kernel = 'linear', C=1.0
The file which was used to build the model, based on a training dataset of 10 sequences, is confusingly called "predictor_all.py".


# FOR PREDICTION:
execute that file: **project/project_TMH/scripts/predictor_only.py**

-two it will use the model that is saved in the output directory as "model.sav"

-it will create an output file in the output directory

-it will show the results in the terminal directly
  
 
