# Face Recognition Application

## Abstract

A Python implementation of Retina + Arcface for detects faces and performs facial/ identity recognition. This application contains two main functions: registering faces and perform recognition on real-time video feed.

## Requirements
- [Install CUDA and CUDNN](https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805) (use VPN if your ISP block Medium)
- Get source code and install python dependencies (using virtual environment is recommend)
```sh
git clone https://github.com/maybehieu/pytorch-face-recognition
pip install -r requirements.txt
```
- Database structure:
>
    database
    ├── Identity_A     # Name of this folder will be name of identity
    │   ├── img1.png   # Image contains ONLY ONE face
    │   ├── img2.jpg         
    │   └── ...              
    ├── Identity_B
    └── ...
>

## Training
- In development...

## Inference
- Run the program
```sh
python main.py -m 0 -dp database
# -m: program mode - 0: register, 1:recognition
# -dp: database path - default: database
```
## References
- [pytorch-retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [arcface-torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

## Citations

```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}

@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{An_2022_CVPR,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={4042-4051}
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10492--10502},
  year={2021}
}
```
