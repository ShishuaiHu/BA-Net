# Code for Kidney PArsing Challenge 2022 (KiPA22)

## Preparing
1. Clone this repo:

```bash
git clone https://github.com/ShishuaiHu/BA-Net.git
cd BA-Net
git checkout kipa22
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
# Preprocess Data
nnUNet_plan_and_preprocess -t 1001 --verify_dataset_integrity
nnUNet_plan_and_preprocess -t 1001 -pl2d None -pl3d ExperimentPlanner3DFabiansResUNet_v21

# ResUNet
nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_4Fold 1001 0 -p nnUNetPlans_FabiansResUNet_v2.1
nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_4Fold 1001 1 -p nnUNetPlans_FabiansResUNet_v2.1 
nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_4Fold 1001 2 -p nnUNetPlans_FabiansResUNet_v2.1
nnUNet_train 3d_fullres nnUNetTrainerV2_ResencUNet_4Fold 1001 3 -p nnUNetPlans_FabiansResUNet_v2.1

# BA-Net
nnUNet_train 3d_fullres BANetV2Trainer_1000Epoch 1001 0
nnUNet_train 3d_fullres BANetV2Trainer_1000Epoch 1001 1 
nnUNet_train 3d_fullres BANetV2Trainer_1000Epoch 1001 2
nnUNet_train 3d_fullres BANetV2Trainer_1000Epoch 1001 3
```

## Inference

```bash
# ResUNet
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER1 -t 1001 -m 3d_fullres -tr nnUNetTrainerV2_ResencUNet_4Fold -p nnUNetPlans_FabiansResUNet_v2.1 --save_npz

# BA-Net
nnUNet_predict -i $TEST_DATA_FOLDER -o $OUTPUT_FOLDER2 -t 1001 -m 3d_fullres -tr BANetV2Trainer_1000Epoch -p nnUNetPlans_FabiansResUNet_v2.1 --save_npz

# Ensemble
nnUNet_ensemble -f $OUTPUT_FOLDER1 $OUTPUT_FOLDER2 -o $OUTPUT_FOLDER
```

## Pretrained Weights

Can be downloaded from [here](https://zenodo.org/record/7030423).

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
@misc{https://doi.org/10.48550/arxiv.2208.13338,
  doi = {10.48550/ARXIV.2208.13338},
  url = {https://arxiv.org/abs/2208.13338},
  author = {Hu, Shishuai and Ye, Yiwen and Liao, Zehui and Xia, Yong},
  title = {Boundary-Aware Network for Kidney Parsing},
  publisher = {arXiv},
  year = {2022},
}
```

### Acknowledgements

- The whole framework is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
