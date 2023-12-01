#bin/sh

root_dir="../../code"
cd ${root_dir}

model_file_path="results/LMTAD/porto/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckptepoch_3_batch_7384.pt"

python eval_porto.py --model_file_path ${model_file_path}