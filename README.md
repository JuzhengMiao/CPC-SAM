# Cross Prompting Consistency with Segment Anything Model for Semi-supervised Medical Image Segmentation (MICCAI 2024)

### Introduction

We provide the codes for CPC-SAM on the ACDC Dataset here.
### Requirements
Please see requirements.txt

### Usage
1. Data preparation:

   Please download the preprocessed dataset provided by https://github.com/HiLab-git/SSL4MIS/ first and put them into the diretory "/data/ACDC/".

   Then, please download the raw data of the ACDC dataset and put related testing cases into the diretory "/data/ACDC_raw". The testing cases and corresponding naming conventions should follow the "test.list" in the diretory "/data/ACDC/" downloaded from https://github.com/HiLab-git/SSL4MIS/. The raw data are used to calculate the 95HD and ASD metrics in mm.

2. Train the model:

   Please prepare a GPU and run the following code. Our training is conducted on an A40 GPU with 46 GB GPU memory.
   ```
   python train.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 20 of the "train.py", the path of the diretory "data/ACDC/" should be provided. (2) Line 21 of the "train.py", the output path should be provided. (3) Line 45 of the "train.py", the path of the pre-trained SAM_B weights should be provided. (4) Line  169 and Line 238 of the "utils.py", the path of the diretory "/data/ACDC_raw" should be provided. (5) Line 45 of the "segment_anything/modeling/prompt_encoder_prompt_class.py", self.num_point_embeddings should be set as the number of classes (including the background) + 2. (6) Line 57 of the "train.py", please provide the experiment name.

3. Test the model:

   Please prepare a GPU and run the following code.
   ```
   python test_mean.py
   ```
   Before running the code, related paths should be set appropriately. (1) Line 76-79, please provide the path of the .csv file, which records the prediction results. (2) Provide the same paths for the same parameters set in the training code. (3) Line 104, please provide the path to save the output results. (4) Line 114, please provide the path of the model to be tested. (5) Line 118, please provide the name for the .csv file.

### Acknowledgement
This code is based on the framework of [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/) and [SAMed](https://github.com/hitachinsk/SAMed). We thank the authors for their codebase.

## Citation
If you find the code useful for your research, please consider starring ⭐ and cite our paper:
```sh
@inproceedings{miao2024cross,
  title={Cross prompting consistency with segment anything model for semi-supervised medical image segmentation},
  author={Miao, Juzheng and Chen, Cheng and Zhang, Keli and Chuai, Jie and Li, Quanzheng and Heng, Pheng-Ann},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={167--177},
  year={2024},
  organization={Springer}
}
```

