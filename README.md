# LM-TAD


### Data Preprocessing

#### Porto Dataset
Download the dataset from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

Unzip the folder to ```$PORTO_ROOT``` where ```$PORTO_ROOT```  is a directory that will contain the porto dataset
run the following (takes about 8 minutes):
```
python preprocess/preprocess_porto.py --data_dir $"{PORTO_ROOT} 
```

#### Pattern of Life dataset

Download the data from https://osf.io/s4bje/

Unzip the folder to ```$POL_ROOT``` where ```$POL_ROOT```  is a directory that will contain the pattern of life dataset
run the following (takes about 25 minutes):
```
python preprocess/preprocess_pol.py --data_dir $"{POL_ROOT} 
```

### Training

#### LM-TAD
To train the LMTAD model, run the following command from the root directory
```
sh /scripts/LMTAD/train.sh
```
You can change the ```dataset``` variable to either ```pol``` or ```porto``` to run the training on respective dataset

#### BASELINES
To train the baseline models, run the following commands from the root directory
```
sh scripts/baselines/train_ae.sh
```
The variables ```dataset``` and ```model_type``` control the dataset and the model to run respectively. The ```model_type``` options are ```dae, vae, gmvae```