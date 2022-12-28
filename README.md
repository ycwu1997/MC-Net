# Mutual Consistency Learning for Semi-supervised Medical Image Segmentation
by Yicheng Wu*, Zongyuan Ge, Donghao Zhang, Minfeng Xu, Lei Zhang, Yong Xia, and Jianfei Cai. 
# Semi-supervised Left Atrium Segmentation with Mutual Consistency Training
by Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai*, and Lei Zhang.

### News
```
<01.07.2022> Our paper entitled "Mutual Consistency Learning for Semi-supervised Medical Image Segmentation" has been accepted by Medical Image Analysis;
```
```
<18.04.2022> We provided our pre-trained models on the LA, Pancreas-CT and ACDC datasets, see './MC-Net/pretrained_pth/';
```
```
<16.04.2022> We released the codes;
```
### Introduction
This repository is for our paper: '[Mutual Consistency Learning for Semi-supervised Medical Image Segmentation](https://doi.org/10.1016/j.media.2022.102530)'. Note that, the MC-Net+ model is named as mcnet3d_v2 in our repository and we also provide the mcnet2d_v1 and mcnet3d_v1 versions, which are similar to the MC-Net model in MICCAI 2021: '[Semi-supervised Left Atrium Segmentation with Mutual Consistency Training](https://doi.org/10.1007/978-3-030-87196-3_28)'.

### Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.2 and Python 3.8.10; All experiments in our paper were conducted on a single NVIDIA Tesla V100 GPU.

### Usage
1. Clone the repo.;
```
git clone https://github.com/ycwu1997/MC-Net.git
```
2. Put the data in './MC-Net/data';

3. Train the model;
```
cd MC-Net
# e.g., for 20% labels on LA
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v2 --labelnum 16 --gpu 0 --temperature 0.1
```
4. Test the model;
```
cd MC-Net
# e.g., for 20% labels on LA
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v2 --exp MCNet --labelnum 16 --gpu 0
```

### Citation
If our MC-Net+ model is useful for your research, please consider citing:

      @inproceedings{wu2021semi,
        title={Semi-supervised left atrium segmentation with mutual consistency training},
        author={Wu, Yicheng and Xu, Minfeng and Ge, Zongyuan and Cai, Jianfei and Zhang, Lei},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={297--306},
        volume={12902},
        year={2021},
        doi={10.1007/978-3-030-87196-3\_28},
        organization={Springer, Cham}
        }
      @article{wu2022mutual,
        title={Mutual Consistency Learning for Semi-supervised Medical Image Segmentation},
        author={Wu, Yicheng and Ge, Zongyuan and Zhang, Donghao and Xu, Minfeng and Zhang, Lei and Xia, Yong and Cai, Jianfei},
        journal={Medical Image Analysis},
        volume={81},
        pages={102530},
        year={2022},
        publisher={Elsevier}
        }

### Acknowledgements:
Our code is adapted from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at 'ycwueli@gmail.com'
