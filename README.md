# MPTransformer
This is the offical repository for MPT ("Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation"), and the code within will be organized and refined later.
### Data Preprocessing
The aortic dataset used in this project is sourced from the [AVT Challenge](https://multicenteraorta.grand-challenge.org).  
The preprocessing pipeline follows the standard procedures used in [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) and [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), ensuring compatibility with state-of-the-art segmentation frameworks.
Set the path to the preprocessed data in the designated location in [`paths.py`](paths.py)([line 29-31 of paths.py](nnunet_mpt/paths.py#L29-L31)).
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
Thanks for the codebase from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) and [VovelMorph](https://github.com/voxelmorph/voxelmorph). If MPT proves useful for your research or applications, we would appreciate it if you could cite our work.
