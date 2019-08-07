## [A Lightweight Convolutional Network for Visual Tracking](https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/LiteCNT-LX.Yang.pdf)

### *Lingxiao Yang*

This repository includes Matlab code for an unpublished tech-report (also included in this repository).

**Claim:** *This report has been rejected by some top conferences. The author is a lazy guy and will not re-submit to any other conferences or journals. But the author-self thought it is a good work that may be helpful for others.*

-------------------------------------------------------------------------------

### Introduction 

#### A state-of-the-art lightweight tracker (around 100 KB)

<p align="center">
<img width="640" height="300" src="https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/speed_auc_modelsize_otb2015_crop.png">
<!-- ![OTB2015](https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/soa_otb100.png) -->
</p>


Most of previous researches in regression trackers mainly explore deep models for feature extraction, and then use sophisticated architectures for online detection. Such systems have many a massive number of trainable parameters, introducing the risk of severe over-fitting. Moreover, the increasingly complex model severely compromises the speed for many practical applications. Recently, several lightweight structures based discriminative correlation filters (DCF) have been proposed for tracking problem, while their performance lags far behind some state-of-the-art trackers. We argue that DCFs often learn a single linear template that cannot well discriminate targets from their surroundings. Furthermore, template updates by linear interpolation in such trackers will include many noisy example, degrading the trained model. In this paper, we present a simple yet effective system, called LiteCNT. Our algorithm only consists of three convolutional layers for the whole tracking process. In addition, a multi-region convolutional operator is introduced for regression output. This idea is simple but powerful as it enables our tracker to capture more details of target object. We further derive an efficient and effective operator to approximate multi-region aggregation. Extensive experiments on five benchmark datasets, including OTB2013, OTB2015, UAV123, LaSOT and VOT2017, illustrate that the proposed method performs well comparable with state-of-the-art trackers. For example, our LiteCNT contains only 100 KB in model size and operates at 50 FPS on a single GPU, while obtaining 65.9\% AUC on OTB2015.

--------------------------------------------------------------------------------------


### Requirements

1. MatConvNet-1.0-beta24 or latest 

2. git clone https://github.com/ZjjConan/LiteCNT.git

### How to run the Code

1. Compile the MatConvNet according to the [website](http://www.vlfeat.org/matconvnet/install/)

2. For *One-Pass-Evaluation*, change the paths to your local paths in `run_LiteCNT_ope.m`. For vot testing, please install the vot official toolkit, and simply copy `vot/tracker_LiteCNT.m` to your VOT workspace.

### Packed Results (New Results with better settings)

We provide our new results on various benchmarks for analysis. For VOT testing, we only run the tracker 5 times (not the official 15 times). Please check the following links:

  [BaiduYun](https://pan.baidu.com/s/1joT5CpoouZd6lAnWHPYImg) Password: 9j3s 

  [OneDrive](https://1drv.ms/u/s!ApDrXo4ZyoJrb3Abh3oEphVPrKE?e=j2aAGn)


### Results on OTB2015

<p align="center">
<img width="560" height="240" src="https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/soa_otb100.png">
<!-- ![OTB2015](https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/soa_otb100.png) -->
</p>

### Results on UAV123

<p align="center">
<img width="560" height="240" src="https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/soa_uav123.png">
<!-- ![OTB2015](https://raw.githubusercontent.com/ZjjConan/LiteCNT/master/paper/soa_otb100.png) -->
</p>

**Note:** This is a little difference to the technique paper. We use 1/10 for gaussian labels generation, and use a smaller learning rate to update the model. Some results are shown as below: 


|	                 |   OTB2015     |  UAV123    |  VOT2017 (A)  | VOT2017 (R)  | VOT2017 (EAO) | VOT2017 (RT-EAO) |
| :----            |    :----:     | :----:     | :----:        |  :----:      | :----:        | :----:           |
| Tech-Report      |    65.9       |  **52.2**  | 0.508         | 1.547        | 0.237         | 0.218            |
| This w AMR       |    **66.2**   |  52.1      | **0.509**     | **1.483**    | **0.248**     | **0.214**        |
| This w/o AMR     |    64.3       |  50.8      | 0.504         | 1.553        | 0.230         | 0.208            |
| CFNet [1]        |    56.3       |  47.7      | 0.473         | 3.467        | 0.126         | 0.107            |
| DCFNet [2]       |    58.0       |  --        | --            | --           | --            | --               |


|                  | Model Size    |  Speed |
| :----            |    :----:     | :----: |
| LiteCNT          |    117 KB     | 50     |
| CFNet [1]        |    76 KB      | 43     |
| DCFNet [2]       |    79 KB      | 91     |

------------------------------------------------


### Contact

Lingxiao Yang

Email: lingxiao.yang717@gmail.com

### Citation
If you find this code useful in your research, please consider cite:

    @misc{LiteCNTYang2019,
      author =       {Lingxiao Yang},
      title =        {[A Lightweight Convolutional Network for Visual Tracking},
      howpublished = {\url{https://github.com/ZjjConan/LiteCNT}},
      year =         {2019}
    }

### Reference
[1] Valmadre J, Bertinetto L, Henriques J, et al. End-to-end representation learning for correlation filter based tracking C]. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 2805-2813.

[2] Wang Q, Gao J, Xing J, et al. Dcfnet: Discriminant correlation filters network for visual tracking [J]. arXiv preprint arXiv:1704.04057, 2017.
