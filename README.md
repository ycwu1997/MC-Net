# Mutual Consistency Learning for Semi-supervised Medical Image Segmentation
by Yicheng Wu, Zongyuan Ge, Donghao Zhang, Minfeng Xu, Lei Zhang, Yong Xia, and Jianfei Cai. 
# Semi-supervised Left Atrium Segmentation with Mutual Consistency Training
by Yicheng Wu, Minfeng Xu, Zongyuan Ge, Jianfei Cai, and Lei Zhang.

### Introduction
This repository is for our paper: '[Mutual Consistency Learning for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/2109.09960.pdf)'. Note that, the MC-Net+ model is named as mcnet3d_v2 in our codes. At the same time, we also provide the mcnet2d_v1 and mcnet3d_v1 similar to the MC-Net model in MICCAI 2021: '[Semi-supervised Left Atrium Segmentation with Mutual Consistency Training](https://doi.org/10.1007/978-3-030-87196-3_28)'.

### Installation
This repository is based on PyTorch 1.8.0, CUDA 11.2 and Python 3.8.10;

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
Our codes is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and also hope our model can promote the relevant research as well.

### Questions
Feel free to contact me at 'ycwueli@gmail.com'
