# LM-TAD

### Python environment
The ```create_venv.sh``` has the instruction on how to create an environment to run the experiments in this repository. 
We used python venv and the code was tested on ```python >= 3.9.9```.

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

### EVALUATION

#### LM-TAD
To evaluate the LMTAD model on the POL dataset, run ```sh scripts/LMTAD/eval_lm.sh```.
To evaluate the LMTAD model on the porto dataset, run ```sh scripts/LMTAD/eval_porto.sh```

#### BASELINES
To evaluate the baseline models on the POL dataset, run ```scripts/baselines/eval_ae.sh```
You need to provide the model path to in the eval_ae.sh file. See an example in that file. 
Adjust the vaiables in the eval_ae.sh file as needed