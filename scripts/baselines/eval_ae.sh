#bin/sh

root_dir="../../code"
cd ${root_dir}

# porto
architecture="gmvae" # dae, vae, gmvae

dataset="pol" # porto, pol, trial0

if [[ "${dataset}" == "porto" ]] ; then
    # # ae porto
    dae="./results/dae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"
    vae="./results/vae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_47.pt"
    gmvae="./results/gmvae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"

    model_file_path="${dae},${vae},${gmvae}"

elif [[ "${dataset}" == "pol" ]]; then
    # ae pattern of life 
    
    dae="./results/dae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"
    vae="./results/vae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_47.pt"
    gmvae="./results/gmvae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"

    model_file_path="${dae},${vae},${gmvae}"

elif [[ "${dataset}" == "trial0" ]]; then
    # ae pattern of life 
   
    dae="./results/dae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"
    vae="./results/vae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_47.pt"
    gmvae="./results/gmvae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"

    model_file_path="${dae},${vae},${gmvae}"

    model_file_path="${dae_smaller},${vae_small},${gmvae_small},${gmvae_bigger}"

fi

# -- CURRENT ---

python eval_ae.py --model_file_path ${model_file_path} --dataset ${dataset}