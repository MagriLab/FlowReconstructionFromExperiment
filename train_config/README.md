# Training Configurations

This project uses [ML Collections](https://github.com/google/ml_collections) to change parameters in when starting training.

## Overview 

All options are defined in `train_config.train_options`.

Four major parts of training can be changed,

1. dataloader ([`dataloader`](./train_options/dataloader.py)) - how a dataset is loaded,
1. taking sensor measurement ([`observe`](./train_options/observe.py)) - where and how the sensors are placed in the dataset to obtain measurements used as the targets of the network.
1. loss function ([`loss_fn`](./train_options/loss_fn.py)) - which loss function is used for training,
1. model ([`select_model`](./train_options/select_model.py)) - which neural network model to use.

The *(`option`)* in brackets are their file names in `train_config.train_options`, and the names used to choose the options when starting training.

### Basic Usage
To use the default options, go to the top directory of this project (**.../FlowReconstructionFromExperiment**) and start training with 
    
    python train.py 

This will train an FF+CNN network using the 2D flow past a triangular cylinder case ($Re=100$) with sensors placed in a grid and a physics-informed loss.   

## How To Use
Configurations are separated into two sets - system set-up, training and Weights&Biases.

### System Options
Here is a list of options and their (type:default).

- `--(no)wandb` (bool:False)<br>
Turn on and off to log experiments to Weights&Biases.
- `--(no)wandb_sweep` (bool:False)<br>
Run script in wandb sweep mode. This option will automatically turn on `--wandb`.
- `--(no)chatty' (bool:False)<br>
More information about the progress of the training will be printed.
- `--debug` (str:None)<br>
Run the named script in debug mode. 
For example, turn on debugging for the main training script with `--debug=main`, turn on debugging for both the main training script and the model subpackage with `--debug=main --debug=flowrec.models`.
- `gpu_id` (int:None)<br>
Which GPU. This affects environment varibale `CUDA_VISIBLE_DEVICES`.
If None, it will use the first available one.
- `gpu_mem` (float:0.3)<br>
Fraction of memory to pre-allocate to the program from the chosen GPU.
- `result_dir` (str:./local_results/)<br>
Path to the directory where the result folder will be saved.
- `result_folder_name` (str:None)<br>
The name of the folder where the results of this particular training will be saved under.
If None, the folder will be named *option codes*+*starting time*.
Option codes are defined in [`train_config.option_codes`](./option_codes.py). 
An example is *t2gpfcpi32304150439548*.

### Setting Training Cases

Select the training case by 

    --cfg train_config/config.py:case@option,case@option

*Case* and *option* are discussed [here](#overview).
All options where some common parameters, some options have specific parameters with no default values.
Some examples are

    --cfg train_config/config.py:observe@grid
    --cfg train_config/config.py:loss_fn@physicswithdata
    --cfg train_config/config.py:loss_fn@physicswithdata,select_model@ffcnn,loss_fn@physicswithdata

