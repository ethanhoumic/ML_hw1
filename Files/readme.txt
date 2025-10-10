This is the readme.txt file of ML assignment 1 Q5.

1. File Structure

Please make sure that the file structure is same as below: 

b11202014/
├── train/
│   ├── log_1D_policy_train.txt
│   ├── log_2D_policy_train.txt
│   ...
│   └── log_9D_policy_train.txt
├── test/
│   ├── 1.txt
│   ├── 2.txt
│   ...
│   └── 458.txt
├── Q5.py
├── lgb_model.pkl
└── readme.txt

Make sure that the train/ and test/ directories contain their .txt files and are in the same directory with Q5.py and lgb_model.pkl. 

2. Evaulation

To evaluate, run Q5.py with the following command:
python .\Q5.py --test_dir test

This will load the pre-trained lgb_model.pkl and produce a submission.csv file. 

3. Training

To train a new model, run Q5.py with the following command:
python .\Q5.py --train

This will train a new ensemble and save it as lgb_model.pkl
(Note: This progress will overwrite the existing lgb_model.pkl file. )