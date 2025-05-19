# MPTransformer
This is the trial-run repository for MPT, and the code within will be organized and refined later.
### Data Preprocessing
The aortic dataset used in this project is sourced from the [SegA Challenge](https://multicenteraorta.grand-challenge.org).  
The preprocessing pipeline follows the standard procedures used in [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) and [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), ensuring compatibility with state-of-the-art segmentation frameworks.
Set the path to the preprocessed data in the designated location in [`paths.py`](paths.py)([line 29-31 of paths.py](https://github.com/iCherishxixixi/MPTransformer/blob/1314c8858eb55cf2a6011d0c8a8c771da917e1b6/nnunet_mpt/paths.py#L29-L31)).
### Set Up the Environment
You can either set up the environment using `setup.py` provided in this project, or directly use the environment configuration from [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt).
### Training and Testing
We provide three versions of the MPT model:

- **3D MPT**: The standard 3D version of the MPT model.
- **3D MPTUNETR**: A hybrid model combining CNN and Transformer architectures.
- **2D MPT**: A 2D model based on an improved version of [TransUNet](https://github.com/Beckschen/TransUNet).

Switch the root directory to [`nnunet_mpt/run`](nnunet_mpt/run), then the training and testing scripts for each model are as follows:

- 3D MPT: [`MPT.sh`](nnunet_mpt/run/MPT.sh)
- MPTUNETR: [`MPTUNETR.sh`](nnunet_mpt/run/MPTUNETR.sh)
- 2D MPT: [`MPT2D.sh`](nnunet_mpt/run/MPT2D.sh)
### Training Weights
Our training weights will be open-sourced soon.
