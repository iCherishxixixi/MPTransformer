import torch
import torch.nn as nn
import sys 
sys.path.append("../network_architecture/custom_modules/custom_networks/UNETRPP/")
from model.unetr_pp import UNETR_PP as UNETRPP_Orig
from nnunet_mpt.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet_mpt.network_architecture.neural_network import SegmentationNetwork
from nnunet_mpt.utilities.nd_softmax import softmax_helper


class UNETRPP(UNETRPP_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['out_channels']
        self.do_ds = False


class nnUNetTrainerV2_UNETRPP(nnUNetTrainerV2_noDeepSupervision):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-5
        # self.oversample_foreground_percent = 1.0
    
    def initialize_network(self):
        self.network = UNETRPP(        
            in_channels=self.num_input_channels,
            out_channels=self.num_classes, 
            img_size=(128, 128, 128),
            feature_size=16,
            hidden_size=256,
            num_heads=4,
            pos_embed="perceptron",  # TODO: Remove the argument
            norm_name="instance",
            dropout_rate=0.0,
            depths=[3, 3, 3, 3],
            dims=[32, 64, 128, 256],
            conv_op=nn.Conv3d,
            do_ds=False,
            )

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None