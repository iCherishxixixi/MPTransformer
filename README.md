# MPTransformer
This is the offical repository for MPT ([Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation](https://arxiv.org/abs/2511.06897)), and the code within will be organized and refined later.
### Data Preprocessing
The aortic datasets used in this project are sourced from [AVT Challenge](https://multicenteraorta.grand-challenge.org), [TBAD Challenge](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad) and [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org).  
The preprocessing pipeline follows the standard procedures used in [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), ensuring compatibility with prevailed segmentation frameworks.
Set the path to the preprocessed data in the designated location in [`paths.py`](nnunet_mpt/paths.py)([line 29-31 of paths.py](nnunet_mpt/paths.py#L29-L31)).
### Set the Environment
You can either set up the environment using `setup.py` provided in this project, or directly use the environment configuration from [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt).
### Training and Testing
We provide three versions of the MPT model:

- **3D MPT**: The standard 3D version of the MPT model.
- **3D MPTUNETR**: A hybrid model combining CNN and Transformer architectures.
- **2D MPT**: A 2D model based on an improved version of [TransUNet](https://github.com/Beckschen/TransUNet). Compared to the original MPT, 2DMPT provides higher-resolution inputs and achieves better Dice scores. However, the lack of inter-slice connectivity for the 2D model poses challenges for maintaining segmentation connectivity.

Switch the root directory to [`nnunet_mpt/run`](nnunet_mpt/run), then the training and testing scripts for each model are as follows:

- 3D MPT: [`MPT.sh`](nnunet_mpt/run/MPT.sh)
- MPTUNETR: [`MPTUNETR.sh`](nnunet_mpt/run/MPTUNETR.sh)
- 2D MPT: [`MPT2D.sh`](nnunet_mpt/run/MPT2D.sh)
### Acknowledgments
Thanks for the codebase from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), [DSCNet](https://github.com/YaoleiQi/DSCNet) and [VovelMorph](https://github.com/voxelmorph/voxelmorph). 

### Citations
If MPT is useful for your research or applications, we would appreciate it if you could cite our work.
```bibtex
@article{zhang2025adaptive,
  title={Adaptive Morph-Patch Transformer for Arotic Vessel Segmentation},
  author={Zhang, Zhenxi and Zheng, Fuchen and Iltaf, Adnan and Han, Yifei and Cheng, Zhenyu and Du, Yue and Li, Bin and Liu, Tianyong and Zhou, Shoujun},
  journal={arXiv preprint arXiv:2511.06897},
  year={2025}
}
```
