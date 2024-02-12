# DSC180B-Popularity Bias in Netflix dataset

## Dataset Download
Netflix: The original data is downloaded from Netflix Kaggle competition published on following [original data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/)\
Specific data formats for the two models are written in the data folders.

# PDA
Causal Inference for de-confounding popularity bias. This is a replicated project from [Causal Intervention for Leveraging Popularity Bias in
Recommendation](https://arxiv.org/pdf/2105.06067.pdf) based on TensorFlow. 

## Requirement 
tensorflow == 1.14 \
Cython (for neurec evaluator)\
Numpy\
prefetch-generator\
python3\
pandas == 0.25.0

To set up the environment and control package versions, please download [environment.yml](PDA/environment.yml) and create the environment `myenv` by entering `conda env create -f environment.yml` in the terminal. \
Activate `myenv` by entering `conda activate myenv` (or `source activate myenv` for DSMLP terminal).\
Then install employed packages in `myenv` by entering:\
`pip install tensorflow==1.14` \
`python -m pip install scipy` \
`pip install prefetch-generator` \
`pip install -U matplotlib` 

After version control, you then can run the code in `myenv`. (Simply activate the environment by `source activate myenv`)

## Parameters
Ket parameters in `simple_reproduce.py` and `train_new_api.py`:\
--pop_exp: gamma in paper (applicable only for PD and PDA models).\
--train: model selection (normal:BPRMF/BPRMF-A | s_condition:PD/PDA | temp_pop:BPR(t)-pop).\
--test: similar to train.\
-- saveID: saved name flag.\
--Ks: list, set top K.\

others: (you can read help, or "python xxx.py --help")\
--dataset: netflix_movie \
--lr: learning rate\
--save_dir: path to find the file of the saved model 

## Command (To produce results)
Note: Before doing anything else, please cd to PDA folder.\
TODO---

## PD/PDA
To train model, run `nohup python -u MF/train_new_api.py --dataset netflix_movie --epoch 1000 --save_flag 1 --log_interval 5 --start 0 --end 10 --step 1 --batch_size 2048 --lr 1e-2 --train s_condition --test s_condition --saveID s_condition --cuda 1 --regs 1e-2 --valid_set valid --pop_exp 0.15  --save_dir save_model/ > output4.out &`. Here popularity is set to be 0.15\
Note: I'm using DSMLP to run the code so the `save_dir` path includes my username. If you want to run it on DSMLP as well, please modify it to your username. If you run on the local computer, you should use the path leading to the PDA folder.
