#bin/sh

root_dir="./code"
cd ${root_dir}

model_file_path="results/LMTAD/porto2/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckptepoch_6_batch_3299.pt"
device="cuda:4"
testing_data_model_file_path="results/LMTAD/porto1/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckptepoch_1_batch_8061.pt"
area_dif="yes"

python eval_porto.py --model_file_path ${model_file_path} --device ${device} --area_dif ${area_dif} --testing_data_model_file_path ${testing_data_model_file_path} 
