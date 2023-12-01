#bin/sh

root_dir="../../code"
cd ${root_dir}

# porto
architecture="gmvae" # dae, vae, gmvae
root="./results/${architecture}/porto"



dataset="pol" # porto, pol, trial0

if [[ "${dataset}" == "porto" ]] ; then
    # # ae porto
    dae_small="./results/dae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"
    dae_bigger="./results/dae/porto/outlier_False_dim_h_1024_dim_z_128_dim_emb_768/ckpt_epoch_12.pt"

    vae_small="./results/vae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_47.pt"
    vae_bigger="./results/vae/porto/outlier_False_dim_h_1024_dim_z_128_dim_emb_768/ckpt_epoch_21.pt"

    gmvae_small="./results/gmvae/porto/outlier_False_dim_h_256_dim_z_64_dim_emb_128/ckpt_epoch_49.pt"
    gmvae_bigger="./results/gmvae/porto/outlier_False_dim_h_1024_dim_z_128_dim_emb_768/ckpt_epoch_38.pt"

    model_file_path="${dae_small},${dae_bigger},${vae_small},${vae_bigger},${gmvae_small},${gmvae_bigger}"

elif [[ "${dataset}" == "pol" ]]; then
    # ae pattern of life 
    
    dae_bigger="./results/dae/pattern_of_life/work-outliers/checkin-atl/outlier_True/features_place/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_21.pt"
    
    vae_bigger="./results/vae/pattern_of_life/work-outliers/checkin-atl/outlier_True/features_place/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_9.pt"
    gmvae_bigger="./results/gmvae/pattern_of_life/work-outliers/checkin-atl/outlier_True/features_place/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_6.pt"

    model_file_path="${dae_bigger}"

elif [[ "${dataset}" == "trial0" ]]; then
    # ae pattern of life 
   
    dae_smaller="./results/dae/trial0/outlier_True/features_gps/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_49.pt"

    
    vae_small="./results/vae/trial0/outlier_True/features_gps/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_49.pt"
    gmvae_small="./results/gmvae/trial0/outlier_True/features_gps/dim_h_128_dim_z_32_dim_emb_64/ckpt_epoch_49.pt"
    gmvae_bigger="./results/gmvae/trial0/outlier_True/features_gps/dim_h_1024_dim_z_128_dim_emb_768/ckpt_epoch_49.pt"

    model_file_path="${dae_smaller},${vae_small},${gmvae_small},${gmvae_bigger}"

fi

# -- CURRENT ---

python eval_ae.py --model_file_path ${model_file_path} --dataset ${dataset}