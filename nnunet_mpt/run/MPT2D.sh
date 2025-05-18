fl_dir="MPT2D"
tr_name="nnUNetTrainerV2_MPT2D"
gpu_id=3
mody="2d"
CUDA_VISIBLE_DEVICES=$gpu_id python run_training.py $mody $tr_name Task618_SegA 0 -p nnUNetPlansv2.1_trgSp_1x1x1
cd ../inference/
CUDA_VISIBLE_DEVICES=$gpu_id  python predict_simple.py -o "./nnUNetv1/Prediction/${fl_dir}" -m $mody -chk model_best -tr $tr_name -i "./nnUNetv1/raw_data_base/nnUNet_raw_data/Task220_SegI/imagesTs/" -t Task618_SegA -f 0 -p nnUNetPlansv2.1_trgSp_1x1x1