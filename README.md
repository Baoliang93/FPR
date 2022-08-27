# FPR
Code for paper "No-Reference Image Quality Assessment by hallucinating Pristine Features".
![image](https://user-images.githubusercontent.com/75255236/121126057-1fbca280-c85a-11eb-9b6d-2d221a83b263.png)


# Environment
* Python python=3.8.5
* pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0

# Running
* Data Prepare
- [x] Download the natural image (NI) datasets and screen content image (SCI) datasets in to the path: `./FPR/datasets/`
- [x] We provide the pretrained checkpoints [here](https://mega.nz/folder/iDxH3R6a#WF25kk1XD30fhlZeSPJzDA). You can download it and put the included  files in to the path: `./FPR/FPR_IQA/FPR_NI/models/" or "./FPR/FPR_IQA/FPR_SCI/models/`. 

* Train: 
  - For NI:  
    `python iqaScrach.py --list-dir='../scripts/dataset_name/' --resume='../models/model_files/checkpoint_latest.pkl' --pro=3 --dataset='dataloader_name'`  
      - dataset_name can be: tid2013, databaserelease2, CSIQ, and kadid10k  
      - model_files  can be: tid2013, live, csiq, and kadid
      - dataloader_name: 'IQA' (for live and csiq  datasets), 'TID2013', and 'KADID' 
 
* Test:  
  `python  ./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py --TrainIndex=1  
  （TrainIndex=1：using the CVD2014 datase as source dataset; 2: LIVE-Qua; 3: LIVE-VQC; 4: KoNviD）`  

# Details
* The model trained on each above four dataset have been provided in "./GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/"
* The code for VGG based feature extraction is available at: https://mega.nz/file/LXhnETyD#M6vI5M9QqStFsEXCeiMdJ8BWRrLxvRbkZ1rqQQzoVuc
* In the intra-dataset setting, it should be noted that we use 80% data for training and the rest 20% data for testing. We haven't used the 20% data for the best epoch selection to avoid testing data leaky, instead, the last epoch is used for performance validation.

