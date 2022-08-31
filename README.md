# Code for Multi-Modality Abdominal Multi-Organ Segmentation Challenge 2022 (AMOS22)

## Preparing
1. Clone this repo:

```bash
git clone https://github.com/ShishuaiHu/BA-Net.git
cd BA-Net
git checkout amos22
```

2. Create experimental environment using virtual env:

```bash
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate

cd nnUNet
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .  # install torch and nnUNet (equipped with BA-Net)
pip install hiddenlayer graphviz IPython
```

3. Configure the paths in `.envrc` to the proper path:

```bash
echo -e '
export nnUNet_raw_data_base="nnUNet raw data path you want to store in"
export nnUNet_preprocessed="nnUNet preprocessed data path you want to store in, SSD is prefered"
export RESULTS_FOLDER="nnUNet trained models path you want to store in"' > .envrc

source .envrc # make the variables take effect
```

## Training
```bash
# Download the dataset and place them in $nnUNet_raw_data_base/nnUNet_raw_data/Task1001_AMOS_CT (only CT) and $nnUNet_raw_data_base/nnUNet_raw_data/Task1002_AMOS_MR (only MR)
# Also, you need to manually place the preprocessed MR and CT in a same folder

# Preprocess Data
nnUNet_plan_and_preprocess -t 1001 --verify_dataset_integrity
nnUNet_plan_and_preprocess -t 1002 --verify_dataset_integrity

# Task 1
## BA-Net-L
nnUNet_train 3d_lowres BANetV2Trainer 1001 0
nnUNet_train 3d_lowres BANetV2Trainer 1001 1 
nnUNet_train 3d_lowres BANetV2Trainer 1001 2
nnUNet_train 3d_lowres BANetV2Trainer 1001 3

## BA-Net-H
nnUNet_train 3d_fullres BANetV2Trainer 1001 0
nnUNet_train 3d_fullres BANetV2Trainer 1001 1 
nnUNet_train 3d_fullres BANetV2Trainer 1001 2
nnUNet_train 3d_fullres BANetV2Trainer 1001 3

# Task 2
## You need to manually merge the MR and CT data and their plan as Task1003
## For Task1002 (MR only)
nnUNet_train 3d_fullres nnUNetTrainerV2 1002 0
nnUNet_train 3d_fullres nnUNetTrainerV2 1002 1 
nnUNet_train 3d_fullres nnUNetTrainerV2 1002 2
nnUNet_train 3d_fullres nnUNetTrainerV2 1002 3

## For Task1003 (MR and CT)
nnUNet_train 3d_fullres BANetV2Trainer 1003 0
nnUNet_train 3d_fullres BANetV2Trainer 1003 1 
nnUNet_train 3d_fullres BANetV2Trainer 1003 2
nnUNet_train 3d_fullres BANetV2Trainer 1003 3
```

## Inference

```bash
# Task 1
## BA-Net-L
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER1 -t 1001 -m 3d_lowres -tr BANetV2Trainer --save_npz

## BA-Net-H
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER2 -t 1001 -m 3d_fullres -tr BANetV2Trainer --save_npz

## Ensemble
nnUNet_ensemble -f $OUTPUT_FOLDER1 $OUTPUT_FOLDER2 -o $OUTPUT_FOLDER

# Task 2
## For CT, the model for task 1 can be used
## For MR, the unified trained model can be used

## CT image inference 
### BA-Net-L
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER1 -t 1001 -m 3d_lowres -tr BANetV2Trainer --save_npz

### BA-Net-H
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER2 -t 1001 -m 3d_fullres -tr BANetV2Trainer --save_npz

### Ensemble
nnUNet_ensemble -f $OUTPUT_FOLDER1 $OUTPUT_FOLDER2 -o $OUTPUT_FOLDER

## MR image
### BA-Net-U
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER1 -t 1003 -m 3d_lowres -tr BANetV2Trainer --save_npz

### nnUNet-M
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER2 -t 1002 -m 3d_lowres -tr BANetV2Trainer --save_npz

### Ensemble
nnUNet_ensemble -f $OUTPUT_FOLDER1 $OUTPUT_FOLDER2 -o $OUTPUT_FOLDER
```

## Pretrained Weights

Can be downloaded from [here](https://zenodo.org/record/7030453).

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:

```
@inproceedings{hu2020boundary,
  title={Boundary-aware network for kidney tumor segmentation},
  author={Hu, Shishuai and Zhang, Jianpeng and Xia, Yong},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={189--198},
  year={2020},
  organization={Springer}
}
@misc{https://doi.org/10.48550/arxiv.2208.13774,
  doi = {10.48550/ARXIV.2208.13774},
  url = {https://arxiv.org/abs/2208.13774},
  author = {Hu, Shishuai and Liao, Zehui and Xia, Yong},
  title = {Boundary-Aware Network for Abdominal Multi-Organ Segmentation},
  publisher = {arXiv},
  year = {2022},
}
```

### Acknowledgements

- The whole framework is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
