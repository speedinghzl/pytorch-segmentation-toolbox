# Pytorch-segmentation-toolbox [DOC](https://weiyc.github.io/assets/pdf/toolbox.pdf)
Pytorch code for semantic segmentation. This is a minimal code to run PSPnet and Deeplabv3 on Cityscape dataset.
Shortly afterwards, the code will be reviewed and reorganized for convenience.

**The new version toolbox is released on branch [Pytorch-1.1](https://github.com/speedinghzl/pytorch-segmentation-toolbox/tree/pytorch-1.1) which supports Pytorch 1.0 or later and distributed multiprocessing training and testing**

### Highlights of Our Implementations
- Synchronous BN
- Fewness of Training Time
- Better Reproduced Performance

### Requirements

To install PyTorch>=0.4.0, please refer to https://github.com/pytorch/pytorch#installation. 

4 x 12g GPUs (e.g. TITAN XP) 

Python 3.6

### Compiling

Some parts of InPlace-ABN have a native CUDA implementation, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

### Training and Evaluation
```bash
./run_local.sh YOUR_CS_PATH
``` 

### Benefits
Some recent projects have already benefited from our implementations. For example, [CCNet: Criss-Cross Attention for semantic segmentation](https://github.com/speedinghzl/CCNet) and [Object  Context  Network(OCNet)](https://github.com/PkuRainBow/OCNet) currently  achieve  the  state-of-the-art  resultson  Cityscapes  and  ADE20K. In  addition, Our code also make great contributions to [Context Embedding with EdgePerceiving (CE2P)](https://github.com/liutinglt/CE2P), which won the 1st places in all human parsing tracks in the 2nd LIP Challange. 

### Citing

If you find this code useful in your research, please consider citing:

    @misc{huang2018torchseg,
      author = {Huang, Zilong and Wei, Yunchao and Wang, Xinggang, and Liu, Wenyu},
      title = {A PyTorch Semantic Segmentation Toolbox},
      howpublished = {\url{https://github.com/speedinghzl/pytorch-segmentation-toolbox}},
      year = {2018}
    }

### Thanks to the Third Party Libs
[inplace_abn](https://github.com/mapillary/inplace_abn) - 
[Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab) - 
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
