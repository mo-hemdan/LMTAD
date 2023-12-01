# LM-TAD


### Data Preprocessing

#### Porto Dataset
Download the dataset from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

Unzip the folder to ```PORTO_ROOT``` where ```PORTO_ROOT```  is a directory that will contain the porto dataset
run the following (takes about 8 minutes):
```
python preprocess/preprocess_porto.py --data_dir $"{PORTO_ROOT} 
```

#### Pattern of Life dataset

Download the data from https://osf.io/s4bje/

Unzip the folder to ```POL_ROOT``` where ```POL_ROOT```  is a directory that will contain the pattern of life dataset
run the following (takes about 25 minutes):
```
python preprocess/preprocess_pol.py --data_dir $"{POL_ROOT} 
```

### Training

#### LM-TAD
##### 