# RescueNet Dataset Source

The RescueNet Dataset is created by Tashnim Chowdhury, Robin Murphy, and Maryam Rahnemoonfar. Their GitHub repository is this [link](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation). Their original README file is here:

------------------------------------------------------------------------------
_(start)

# RescueNet Dataset 

## Overview

Frequent, and increasingly severe, natural disasters threaten human health, infrastructure, and natural systems. The provision of accurate, timely, and understandable information has the potential to revolutionize disaster management. For quick response and recovery on a large scale, after a natural disaster such as a hurricane, access to aerial images is critically important for the response team. The emergence of small unmanned aerial systems (UAS) along with inexpensive sensors presents the opportunity to collect thousands of images after each natural disaster with high flexibility and easy maneuverability for rapid response and recovery.  Moreover, UAS can access hard-to-reach areas and perform data collection  tasks that can be unsafe for humans if not impossible.  Despite all these advancements and efforts to collect such large datasets, analyzing them and extracting meaningful information remains a significant challenge in scientific communities.

RescueNet provides high-resolution UAS imageries with detailed semantic annotation regarding the damages.

![alt text](https://github.com/tashnimchowdhury/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/blob/main/example-rescuenet-all-cls.PNG?raw=true)


## Dataset Details

The data is collected with a small UAS platform, DJI Mavic Pro quadcopters, after Hurricane Michael. The whole dataset has 4494 images, divided into training (~80%), validation (~10%), and test (~10%) sets. The semantic segmentation labels include: 1) Background, 2) Debris, 3) Water, 4)Building No Damage, 5) Building Minor Damage, 6) Building Major Damage, 7) Buidling Total Destruction, 8) Road, 9)Vehicle, 10) Tree, 11) Pool, 12) Sand. 

The dataset can be downloaded from this link: https://drive.google.com/drive/folders/1XNgPVmiu9egr1fywgNeXfnxojFOe_INT?usp=sharing

## License

This dataset is released under the [Community Data License Agreement (permissive)](https://cdla.io/permissive-1-0/).

### Paper Link
The paper can be downloaded from this [link](https://arxiv.org/pdf/2202.12361.pdf).
Please cite our paper when using the dataset

 ```
 
@article{chowdhury2022rescuenet,
  title={RescueNet: A High Resolution UAV Semantic Segmentation Benchmark Dataset for Natural Disaster Damage Assessment},
  author={Chowdhury, Tashnim and Murphy, Robin and Rahnemoonfar, Maryam},
  journal={arXiv preprint arXiv:2202.12361},
  year={2022}
}

```
_(End)_

------------------------------------------------------------------------------

# Implementation

The dataset is first downloaded into Google Colab, unzipped in the Colab notebook, and then reassembled into 13 smaller folders for easier download. Each folder is then uploaded onto Kaggle. 

1. rescuenet-train-1-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-1-of-10/)
2. rescuenet-train-2-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-2-of-10/)
3. rescuenet-train-3-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-3-of-10/)
4. rescuenet-train-4-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-4-of-10/)
5. rescuenet-train-5-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-5-of-10/)
6. rescuenet-train-6-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-6-of-10/)
7. rescuenet-train-7-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-7-of-10/)
8. rescuenet-train-8-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-8-of-10/)
9. rescuenet-train-9-of-10 (360 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-9-of-10/)
10. rescuenet-train-10-of-10 (355 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-10-of-10/)
11. rescuenet-train-missed (733 images, 0 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-missed) # 733 images was downloaded twice during the split-download of the 3595 train images, and therefore 733 are lost. Here the missed images are re-collected.

12. rescuenet-train-labels (0 images, 3595 labels) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-train-labels)

13. rescuenet-val (449 images-label pairs) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-val)

14. rescuenet-test (450 images-label pairs) [link](https://www.kaggle.com/datasets/royzhao2004/rescuenet-test)
