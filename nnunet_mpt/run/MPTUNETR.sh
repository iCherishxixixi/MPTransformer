fl_dir="MPTUNETR"
tr_name="nnUNetTrainerV2_MptUNETR_128x128x128"
gpu_id=1
CUDA_VISIBLE_DEVICES=$gpu_id python run_training.py 3d_fullres $tr_name Task618_SegA 0 -p nnUNetPlansv2.1_trgSp_1x1x1
cd ../inference/
CUDA_VISIBLE_DEVICES=$gpu_id python predict_simple.py -chk "model_best" -o "./nnUNetv1/Prediction/${fl_dir}" -m 3d_fullres -tr $tr_name -i "./nnUNetv1/raw_data_base/nnUNet_raw_data/Task618_SegA/imagesTs/" -t Task618_SegA -f 0 -p nnUNetPlansv2.1_trgSp_1x1x1