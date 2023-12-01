#bin/sh

root_dir="./code"
cd ${root_dir}

# porto
architecture="gmvae" # dae, vae, gmvae

dataset="pol" # porto, pol, trial0

if [[ "${dataset}" == "porto" ]] ; then
    # # ae porto
    dae=""
    vae=""
    gmvae=""

    model_file_path="${dae},${vae},${gmvae}"

elif [[ "${dataset}" == "pol" ]]; then
    # ae pattern of life 
    
    dae=""
    vae=""
    gmvae="results/gmvae/pol/work-outliers/checkin-atl/outlier_True/features_place/dim_h_1024_dim_z_128_dim_emb_768/ckpt_epoch_49.pt"

    model_file_path="${dae},${vae},${gmvae}"

fi

# -- CURRENT ---

python eval_ae.py --model_file_path ${model_file_path} --dataset ${dataset}