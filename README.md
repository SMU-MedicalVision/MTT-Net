# MTT-Net: Multi-scale Tokens-Aware Transformer Network for Multi-region and Multi-sequence MR-to-CT Synthesis in A Single Model
## Experiments
- CUDA/CUDNN
- torch >=1.12.0
- timn >=0.5.4
- numpy >=1.22.3

## Dataset
We applied N4 bias field correction to the data and performed registration between MR and CT images. The paired data is stored in the following format:
```
/Datasets/
    ├──Headneck_001
      ├── MR.nii.gz
      ├── CT.nii.gz
      ├── mask.nii.gz
    ├──Headneck_002
      ├── MR.nii.gz
      ├── CT.nii.gz
      ├── mask.nii.gz
    .
    .

    ├──Abdomen_001
      ├── MR.nii.gz
      ├── CT.nii.gz
      ├── mask.nii.gz
    ├──Abdomen_002
      ├── MR.nii.gz
      ├── CT.nii.gz
      ├── mask.nii.gz
```

## Train
To run the train.py file, you need to set common parameters such as the data storage path and patch size. If you need to use VGG perceptual loss, you can go to the official website and download the pre-trained model of VGG19: vgg19-dcbb9e9d.pth.