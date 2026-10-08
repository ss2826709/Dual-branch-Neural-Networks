# Dual-branch Neural Networks with Emission-source-wise Corrections for PM2.5 Estimation

## Overview

Most machine learning models for Chemical Transport Model bias correction output a single total PM2.5 concentration and discard source contribution information. They also process heterogenous input features on the same pathway despite their different data structure. This project develops neural network architectures that process input features in dual path and retain source contribution information in the final output.

## Model components
1. Backbone Architectures: this component determines how the input features are processed and transformed into hidden representations
    - Multilayer Perceptron (MLP)
    - 1D convolutional network (CNN1D)
    - Dual-branch (DB, the proposed architecture): Combines two parallel branches, an MLP branch for static inputs and a CNN1D branch for time-series inputs. The outputs of the two branches are then combined to form the hidden representation.
2. Bias-correction strategy: this component determines how the hidden representation is used to generate the final PM2.5 estimate
    - Single-stage (SS): Directly maps the hidden representation to the final PM2.5 prediction through an output layer.
    - Two-stage (TS): Uses the hidden representation to generate source-specific residual corrections for the CMAQ source contributions. Each source-specific correction head combines the shared hidden representation with its corresponding original CMAQ source contribution and predicts a residual correction. The corrected source contributions are then summed to obtain the final PM2.5 prediction.

## Model summary
1. SS-MLP
2. SS-CNN1D
3. SS-DB
4. TS-MLP
5. TS-CNN1D
6. TS-DB (the proposed model)

## Repository Structure
```
├── functions/       # Missing value imputation, data splitting, model definitions, training
├── hp_tuning/
│   ├── region_holdout/   # Hyperparameter tuning with data split by region
│   └── station_holdout/  # Hyperparameter tuning with data split by station
└── final_eval/      # Final evaluation of all models on the test set using the best configuration from tuning
```
## Software requirements

The code was developed and tested with Python 3.11.

### Conda

Create the Conda environment using:

```bash
conda env create -f environment.yml -n <environment_name>
```

Replace `<environment_name>` with your preferred environment name.

Activate the environment using:

```bash
conda activate <environment_name>
```

### pip

Alternatively, install the required Python packages using:

```bash
pip install -r requirements.txt
```
