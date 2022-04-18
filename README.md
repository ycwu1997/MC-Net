# Mutual Consistency Learning for Semi-supervised Medical Image Segmentation
by Yicheng Wu, Zongyuan Ge, Donghao Zhang, Minfeng Xu, Lei Zhang, Yong Xia, and Jianfei Cai. 
# Semi-supervised Left Atrium Segmentation with Mutual Consistency Training
by Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai, and Lei Zhang.

### News

<18.04.2022> We provided our pre-trained models on the LA, Pancreas-CT and ACDC datasets, see 'MC-Net/pretrained';
<16.04.2022> We uploaded the codes;

### Introduction
This repository is for our paper: '[Mutual Consistency Learning for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/2109.09960.pdf)'. Note that, the MC-Net+ model is named as mcnet3d_v2 in our codes. At the same time, we also provide the mcnet2d_v1 and mcnet3d_v1 similar to MC-Net in MICCAI 2021: '[Semi-supervised Left Atrium Segmentation with Mutual Consistency Training](https://doi.org/10.1007/978-3-030-87196-3_28)'.

### Installation
This repository is based on PyTorch 1.8.0, CUDA 11.2 and Python 3.8.10;

### Usage
1. Clone the repo.;
```
git clone https://github.com/ycwu1997/MC-Net.git
```
2. Put the data in 'MC-Net/data';

3. Train the model
```
e.g.
cd MC-Net
# for 10% labels
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v2 --labelnum 8 --gpu 0 --temperature 0.1
```
4. Test the model
```
e.g.
cd MC-Net
# for 10% labels
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v2 --exp MCNet --labelnum 16 --gpu 0
```

### Citation
If our MC-Net+ model is useful for your research, please consider citing:

      @inproceedings{wu2021semi,
        title={Semi-supervised left atrium segmentation with mutual consistency training},
        author={Wu, Yicheng and Xu, Minfeng and Ge, Zongyuan and Cai, Jianfei and Zhang, Lei},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={297--306},
        year={2021},
        organization={Springer}
        }
      @article{wu2021enforcing,
        title={Enforcing Mutual Consistency of Hard Regions for Semi-supervised Medical Image Segmentation},
        author={Wu, Yicheng and Ge, Zongyuan and Zhang, Donghao and Xu, Minfeng and Zhang, Lei and Xia, Yong and Cai, Jianfei},
        journal={arXiv preprint arXiv:2109.09960},
        year={2021}
        }

### Acknowledgements:
Our codes is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at 'ycwueli@gmail.com'
