#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet_mpt.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_warmup(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1050

    def maybe_update_lr(self, epoch=None):
        if self.epoch < 50:
            # epoch 49 is max
            # we increase lr linearly from 0 to initial_lr
            lr = (self.epoch + 1) / 50 * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr:", lr)
        else:
            if epoch is not None:
                ep = epoch - 49
            else:
                ep = self.epoch - 49
            assert ep > 0, "epoch must be >0"
            return super().maybe_update_lr(ep)
