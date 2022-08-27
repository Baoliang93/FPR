# FPR
Code for paper "No-Reference Image Quality Assessment by hallucinating Pristine Features".
![image](https://github.com/Baoliang93/FPR/blob/main/FPR_IQA/framework.png =542x1000)


# Environment
* Python python=3.8.5
* pytorch=1.7.1=py3.8_cuda11.0.221_cudnn8.0.5_0

# Running
* Data Prepare
- [x] Download the natural image (NI) datasets and screen content image (SCI) datasets in to the path: `./FPR/datasets/`
- [x] We provide the pretrained checkpoints [here](https://mega.nz/folder/iDxH3R6a#WF25kk1XD30fhlZeSPJzDA). You can download it and put the included  files in to the path: `./FPR/FPR_IQA/FPR_NI/models/" or "./FPR/FPR_IQA/FPR_SCI/models/`. 

* Train: 
  - For NI:  
    `python ./FPR/FPR_IQA/FPR_SCI/src/iqaScrach.py --list-dir='../scripts/dataset_name/' --resume='../models/model_files/checkpoint_latest.pkl' --pro=split_id --dataset='dataloader_name'`  
      -    dataset_name: "tid2013", "databaserelease2", "CSIQ", or "kadid10k"  
      -    model_files: "tid2013", "live", "csiq", or "kadid"
      - dataloader_name: "IQA" (for live and csiq  datasets), "TID2013", or "KADID"  
      - split_id: '0' to '9'
  - For SCI:   
      -  SIQAD: `python ./FPR/FPR_IQA/FPR_SCI/srciqaScrach.py  --pro=split_id`    
      -  SCID: `python ./FPR/FPR_IQA/FPR_SCI/srcscid-iqaScrach.py  --pro=split_id`   
      
* Test:  
  - For NI:   
  `python ./FPR/FPR_IQA/FPR_SCI/src/iqaTest.py --list-dir='../scripts/dataset_name/' --resume='../models/model_files/model_best.pkl' --pro=split_id  --dataset='dataloader_name'`  
   - For SCI:   
      -  SIQAD: `python ./FPR/FPR_IQA/FPR_SCI/srciqaTest.py  --pro=split_id`    
      -  SCID: `python ./FPR/FPR_IQA/FPR_SCI/srcscid-iqaTest.py  --pro=split_id`
  

# Details
* Waitting...

