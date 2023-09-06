 # UNNT (Utility for comparing Neural Networks and Tree-based models)

UNNT enables users with structured (tabular) data to compare CNN and XGBoost models by providing a utility to train both models

### Installation

UNNT requires [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) to run. 

#### Create conda environment
```conda env create -f gpu_environment.yml -n UNNT_gpu```

The CPU version can be installed using the environment.yml file

```conda env create -f environment.yml -n UNNT```


### Usage  

Default run configuration trains XGBoost and CNN models with NCI60 datasets

```bash
cd UNNT
python3 unnt.py
```

Use ```--gpu``` when running on ```UNNT_GPU``` conda environment with a GPU

```bash
python3 unnt.py --gpu
```


### Data

All data used to train models with UNNT must be place in the ```data``` folder at the root of this repository

The data necessary to run the in default settings using NCI60 datasets are also located in the ```data``` folder

#### Training with custom data

When training XGBoost and CNN using your dataset, place it in the ```data``` folder at the root of this repository and update **[tree_config.txt](/UNNT/tree_config.txt)** file in ```UNNT``` folder. Provide the name of the file for the configuration parameter **`data_file`**. Name of data file must be in quotes and is case sensitive. The dataset provided is required to be in csv format.

When providing custom data make sure the data is cleaned and preprocessed. The columns must be features. Provide the **target variable** the models train on by setting the **`target_variable`** configuration parameter in **[tree_config.txt](/UNNT/tree_config.txt)**


### Configuration

In addition to configuration on data necessary for the entire software, **[tree_config.txt](/UNNT/tree_config.txt)** also contains XGBoost model parameters.

#### XGBoost parameters

```
n_estimators=500
max_depth=10
eta=0.1
subsample=0.5
colsample_bytree=0.8
```

#### CNN parameters

**[cnn_config.txt](/UNNT/cnn_config.txt)** contains the configurations specific for CNN model. Many of these are specific to NCI60 dataset used for demo in this software and thus won't affect the models trained using custom data.

The main parameters for CNN include

```
dense=[1000, 500, 100, 50]
batch_size=100
epochs=1
activation='tanh'
loss='mse'
optimizer='sgd'
learning_rate=0.01
scaling='std'
dropout=0.1
feature_subsample=0
val_split=0.1
test_split=0.1
```


### The MIT License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Distributed under the MIT License. See [LICENSE](/LICENSE) for more information  
