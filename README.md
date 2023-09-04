# UNNT (Utility for comparing Neural Networks and Tree-based models)

### Installation

UNNT requires Anaconda to run

#### Create conda environment
```conda env create -f environment.yml -n UNNT_gpu```

If the system you will be running on does not have GPUs install the CPU version

```conda env create -f environment.yml -n UNNT```


### Usage  

Default run configuration trains XGBoost and CNN models with NCI60 datasets

```
cd UNNT
python3 unnt.py
```
