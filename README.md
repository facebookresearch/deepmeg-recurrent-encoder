# DeepMEG-Encoding project

This project investigates models for forecasting MEG data, using past MEG and external stimuli. 
These models range from linear to nonlinear, with or without access to the initial brain state, and conclude with our Deep Recurrent Encoder (DRE) architecture.
The DRE outperforms current methods and trains across subjects simulatenously. 
An ablation study yields insight into the modules which best explain its predictive performance. 
A simple feature importance analysis helps interpret what the deep architecture learns.

Predictive Performance             |  Feature Importance
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/37180957/109517969-1c22be00-7aaa-11eb-9511-7301c27bf0ac.png)  |  ![](https://user-images.githubusercontent.com/37180957/109518451-89ceea00-7aaa-11eb-8124-cbaeec97d29c.png)


## General information

You will need Python >= 3.7 to use this code.

Install Python package requirements with:
```
pip install -r requirements.txt
```

Find help with:
```
python3 -m neural --help
```
 
## Data extraction

First, install the MOUS dataset from https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236?3
To do so, you may register via your orchid account.

To extract MEG and stimuli from the MOUS dataset:

```
python3 -m neural.extraction --data /path/to/dataset --out /path/to/extraction
```

**Notes:**
 - This script will create the folder `/path/to/extraction/full`.
 - This step takes a few hours and requires in the extraction folder at least 90GB of disk space.
 - This step does not need a GPU to run.
 - Setting `--n-subjects 1` would perform the extraction over one subject only. This is useful
 if you have a limited disk space and you want to test quickly.
 - Setting `--use-pca` would extract 40 components from the MEG and the extraction folder becomes `/path/to/extraction/40`. This will require less disk space (about 15GB).

Then, proceed to train encoding models.

## Train the Deep Recurrent Encoder (DRE)

To train the DRE:

```
python3 -m neural --data /path/to/extraction/full --out /path/to/metrics
```

The ablations of the DRE (resp. "NO-CONV", "PCA", "NO-SUB", "NO-INIT") in the paper were trained using:
```
python3 -m neural --data /path/to/extraction --out /path/to/metrics --epochs 40 --conv-layers=0
python3 -m neural --data /path/to/full/extraction --out /path/to/metrics --pca 40
python3 -m neural --data /path/to/full/extraction --out /path/to/metrics --subject-dim=0
python3 -m neural --data /path/to/full/extraction --out /path/to/metrics --meg-init 0
```

## Train the Linear Encoders (TRF, RTRF)

```
python3 -m neural.linear --with-forcing --with-init --shuffle --out /path/to/metrics
```

## License

This repository is released under the CC-BY-NC 4.0. license as found in the [LICENSE](LICENSE) file.

