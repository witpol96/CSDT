## CSDT: Cross-Modal Semantic Decoupling and Transfer for Text-to-Visible-Infrared Person Re-Identification
Official PyTorch implementation for Cross-Modal Semantic Decoupling and Transfer for Text-to-Visible-Infrared Person Re-Identification

## Abstract
Text-to-Image Person Re-Identification (TI-ReID) retrieves visible pedestrian images using text queries. Yet in low-light or nighttime settings, visible images lack sufficient identity details, while infrared images effectively capture pedestrian contours and textures. To enable all-day surveillance, we propose a dual cross-modal retrieval task called Text-to-Visible-Infrared Re-Identification (TVI-ReID) and construct corresponding tri-modal datasets. Compared to TI-ReID, TVI-ReID faces two key challenges: (1) complex hybrid discrepancies in dual cross-modal retrieval from three modalities, and (2) semantic inconsistency between pretraining and downstream tasks. To address these issues, we propose a Cross-Modal Semantic Decoupling and Transfer (CSDT) framework. CSDT constructs color-related and color-irrelevant feature subspaces via Semantic Decoupling Learning (SDL) to align shared semantics across text and dual image modalities, reducing hybrid discrepancies. Moreover, Semantic Distribution Transfer (SDT) adapts pretrained text-visible alignment to text-infrared matching. Extensive experiments on tri-modal datasets show our approach outperforms existing state-of-the-art TI-ReID methods.

## Getting Started
#### 1. Clone the Repo
~~~
git clone https://github.com/witpol96/CSDT.git
cd CSDT
~~~
#### 2. Install Dependencies
~~~
conda create -n CSDT python=3.10
pip3 install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
~~~
#### 3. Prepare Datasets and Pretrained Model
Download the SYSU-MM01 and ORBench datasets first. Afterwards, integrate the annotations of SYSU-TVI into the SYSU-MM01 directory, and arrange all datasets following the specified file structure below.
~~~
.
├── SYSU-MM01
│   ├── cam1
│   ├── cam2
│   ├── cam3
│   ├── cam4
│   ├── cam5
│   ├── cam6
│   └── data_captions.json
├── ORBench
│   ├── vis
│   ├── nir
│   ├── ...
│   └── llcm_data_captions.json
...
~~~
Download the pre-trained model from HAM and store it in an appropriate directory.

## Training
Modify the path-related parameters in `scripts/run_SYSU.sh` and `scripts/run_LLCM.sh`, then run the scripts.
~~~
bash scripts/run_SYSU.sh  #train on SYSU-TVI
bash scripts/run_LLCM.sh  #train on LLCM-TVI
~~~

## Contact
2021302111058@whu.edu.cn