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

## Setting Training Cases

Cases are selected by two sets of configurations - training and Weights&Biases.

Select the training case by 

    --cfg train_config/config.py:case@option,case@option

*Case* and *option* are discussed in [Overview](#overview).
Some examples are

    --cfg train_config/config.py:observe@grid
    --cfg train_config/config.py:loss_fn@physicswithdata
    --cfg train_config/config.py:loss_fn@physicswithdata,select_model@ffcnn


If using Weights&Biases, then the case will also need to be set for `wandbcfg` with 

    --wandbcfg train_config/config_wandb.py:model_option,loss_fn_option

For example, 

    --wandbcfg train_config/config_wandb.py:ffcnn,physicswithdata

The cases given for both `cfg` and `wandbcfg` should match.


## System Options
Here is a list of options and their (type:default).

- `--(no)wandb_sweep` (bool:False)<br>
    Run script in wandb sweep mode. This option will automatically turn on `--wandb`.
- `--(no)chatty` (bool:False)<br>
    More information about the progress of the training will be printed.
- `--debug` (str:None)<br>
    Run the named script in debug mode. 
    For example, turn on debugging for the main training script with `--debug=main`, turn on debugging for both the main training script and the model subpackage with `--debug=main --debug=flowrec.models`.
- `--gpu_id` (int:None)<br>
    Which GPU. This affects environment varibale `CUDA_VISIBLE_DEVICES`.
    If None, it will use the first available one.
- `--gpu_mem` (float:0.3)<br>
    Fraction of memory to pre-allocate to the program from the chosen GPU.
- `--result_dir` (str:./local_results/)<br>
    Path to the directory where the result folder will be saved.
- `--result_folder_name` (str:None)<br>
    The name of the folder where the results of this particular training will be saved under.
    If None, the folder will be named *option codes*+*starting time*.
    Option codes are defined in [`train_config.option_codes`](./option_codes.py). 
    An example is *t2gpfcpi32304150439548*.
- `--print` (int:200)<br>
    How often to print the losses to screen. 

**Note: If running in sweep mode, then the user must update the [sweep pre-processing function](./sweep_process_config.py)


## Common Options
All options have some common parameters, some options have specific parameters with no default values.
Common options are the options that are available for all cases. 


### Main Config (`--cfg`)

`cfg`, the main configuration, has four sections, case (see [Setting Training Cases](#setting-training-cases)), data_config, train_config and model_config.
For example, data_dir in data_config is set by `--cfg.data_config.data_dir=path`.

#### data_config

- `data_dir` (str:None)<br>
    Directory where the data is stored.
- `shuffle` (bool:False)<br>
    Whether to shuffle the snapshots before splitting into training/validation/testing sets.
- `randseed` (int:None)<br>
    Random seed to use in `shuffle`, if None but `shuffle` is True, then a new one is generated.
- `remove_mean` (bool:False)<br>
    Whether to centre the dataset around $0$ by subtracting mean.
- `normalise` (book:False)<br>
    Whether to normalise the dataset before training. 
    The data and model prediction will be unnormalised before taking any physics-informed loss.
- `nsample` (int:None)<br>
    How many samples to load. This is then split into the training and validation set.
- `batch_size` (int:None)<br>
    Number of snapshots per batch.
- `re` (float:100.0)<br>
    Reynolds number of the dataset.
- `dt` (float:0.125)<br>
    Time elapsed between two consecutive snapshots.
- `dx` (float:12/512)<br>
    Distance between grids in x direction.
- `dy` (float:4/128)<br>
    Distance between grids in y direction.
- `dz` (float:None)<br>
    Distance between grids in z direction.
- `snr` (float:0.0)<br>
    Signal-to-noise ratio in dB. This adds white noise to the measurement after the data is cropped and shuffled, but before other pre-processing.
- `filter` (str:None)<br>


#### model_config

- `dropout_rate` (float:0.0)
- `name` (str:None) \
    Model name.
- `activation` (str:tanh) \
    We will look up the name of this function in `jax.nn`


#### train_config

- `learning_rate` (float:3e-4)
- `regularisation_strength` (float:0.0)
- `epoches` (int:20000)


### Weights&Biases (`--wandbcfg`)

- `mode` (online | disabled | offline) 
- `use_artifact` (str:None) \
    The name of the artifact.
- `config.log_frequency` (int:10) \
    How many epochs between logging values to Weights&Biases.

For the options to pass to `wandb.init`, see [documentation](https://docs.wandb.ai/ref/python/init).
`Tuple[str]` is used instead of `List[str]`.

If `wandbcfg.name` is not set by user, it will be set to the `result-folder-name`.
Default mode is set to *online* and default project is *FlowReconstruction*.



## Options for Each Case


### Dataloader
--------------------
Options are under **data_config**.

#### 2dtriangle
Generated 2D wake behind a triangle.

- `slice_to_keep` (tuple: ((None,), (None,), (None,250,None), (None,))) <br>
    Define the area to take from the simulation domain.
- `data_dir` (str: ./local_data/re100/) <br>
    Path to the data.
- `re` (float: 100.0) <br>
    Reynolds number used in the simulation.
- `dt` (float: 0.125) <br>
    Time step between snapshots.
- `dx` (float: 12/512) <br>
    Grid spacing in the x-direction.
- `dy` (float: 4/128) <br>
    Grid spacing in the y-direction.
- `pressure_inlet_slice` (tuple: ((0,1,None), (49,80,None))) <br>
    Slice defining the region where pressure inlet values are extracted.
- `nsample` (int: 700) <br>
    Number of simulation samples to load.

#### 2dkol/3dkol
2D/3D Kolmogorov flow. Both 2dkol and 3dkol have the same config options, the defaults shown here are for 3dkol.

- `data_dir` (str: ./local_data/kolmogorov/dim3_re34_k32_f4_dt01_grid64_189.h5) <br>
    Path to the dataset file containing simulation data.
- `re` (float: 34.0) <br>
    Reynolds number used in the simulation.
- `dt` (float: 0.1) <br>
    Time step between snapshots.
- `dx` (float: $2\pi$/64) <br>
    Grid spacing in the x-direction.
- `dy` (float: $2\pi$/64) <br>
    Grid spacing in the y-direction.
- `dz` (float: $2\pi$/64) <br>
    Grid spacing in the z-direction. *This option is not available for 3dkol.*
- `crop_data` (tuple: ((None,), (None,), (None,))) <br>
    Defines spatial cropping bounds for the simulation data along (x, y, z).
- `pressure_inlet_slice` (tuple: ((None,), (0,1,None), (None,))) <br>
    Slice defining the pressure inlet region.
- `forcing_frequency` (int: 4) <br>
    Frequency of the external forcing applied in the simulation.
- `nsample` (int: 900) <br>
    Number of simulation samples to load.


#### 3dkolsets
Multiple decorrelated segments of 3D kolmogorov flow.

- `data_dir` (str: ./local_data/kolmogorov/dim3_datasets_080325.txt) <br>
    Path to the dataset file containing metadata.
- `re` (float: 34.0) <br>
    Reynolds number used in the simulation.
- `dt` (float: 0.1) <br>
    Time step between simulation frames.
- `dx` (float: $2\pi$/64) <br>
    Grid spacing in the x-direction.
- `dy` (float: $2\pi$/64) <br>
    Grid spacing in the y-direction.
- `dz` (float: $2\pi$/64) <br>
    Grid spacing in the z-direction.
- `crop_data` (tuple: ((None,), (None,), (None,))) <br>
    Defines spatial cropping bounds for the simulation data along (x, y, z).
- `pressure_inlet_slice` (tuple) <br>
    Placeholder slice defining the pressure inlet region (to be specified).
- `random_input` (tuple) <br>
    Placeholder tuple containing (random seed, number of pressure sensors).
- `forcing_frequency` (int: 4) <br>
    Frequency of the external forcing applied in the simulation.
- `nsample` (int: 2500) <br>
    Number of simulation samples to load.


### Observe
------------------------
Options are under **data_config**.

#### grid/grid_pin

Sensors are placed in a grid. / Sensors are placed in a grid, with extra pressure sensors.

- `slice_grid_sensors` (tuple: ((None,None,15), (None,None,5)))<br>
    Define a grid with (start,end,step).
- `components` (str: all) <br>
    Which components to taken measurements from. 
    - 'all': all components
    - 'velocity': all velocities, equivalent to slice the last dimension of the data using [\:-1]
    - 'ij...': component with index i, j, ...

#### sparse/sparse_pin
Sensors are placed sparsely in the domain. / Sensors are placed sparsely in the domain, with extra pressure sensors.

**data_config**

- `sensor_index` (tuple: None)<br>
    Coordinates of the sensors ((x1,x2,...),(y1,y2,...)).
    Can also be 3D.
- `components` (str: all) <br>
    Which components to taken measurements from. 
    - 'all': all components
    - 'velocity': all velocities, equivalent to slice the last dimension of the data using [\:-1]
    - 'ij...': component with index i, j, ...

#### random_pin
Sensors are randomly placed in the domain.

- `random_sensors` (tuple: None)<br>
    (random seed, number of sensors).
- `sensor_index` (tuple: None)<br>
    Coordinates of the sensors ((x1,x2,...),(y1,y2,...)).
    Can also be 3D.
- `components` (str: all) <br>
    Which components to taken measurements from. 
    - 'all': all components
    - 'velocity': all velocities, equivalent to slice the last dimension of the data using [\:-1]
    - 'ij...': component with index i, j, ...

#### slice/slice_pin
Measurement planes are taken at indexes listed in xplane, yplane and zplane.

- `xplane` (str: "")<br>
    "x1,x2..." each x is an integer index.
- `yplane` (str: "")<br>
    "y1,y2..." each y is an integer index.
- `zplane` (str: "32")<br>
    "z1,z2..." each z is an integer index.
- `components` (str: "velocity") <br>
    Which components to taken measurements from. Inorder for the planes at x1,x2...y1,y2...z1,z2...
    - 'all': all components
    - 'velocity': all velocities, equivalent to slice the last dimension of the data using [\:-1]
    - 'ij...': component with index i, j, ...


### Select_model
---------------------------------
Options are under **model_config**

#### ffcnn
Fully connected layers followed by convolution layers.
For details of the network and the arguments it takes, see [here](../flowrec/models/cnn.py). 

- `mlp_layers` (tuple:(96750,))<br>
    Number of neurons in (layer1,layer2,layer3...).
- `output_shape` (tuple:(250,129,3))<br>
    Expected shape of the output.
- `cnn_channels` (tuple:(32,16,3))<br>
    Number of channels per convolution layer.
    last number should correspond to the last number in output_shape.
- `cnn_filters` (tuple:((3,3),))<br>
    Size of the convolution filters.

#### ff
Simple feedforward network.

#### fc2branch
Dual-branch CNN network, with an optional Fourier Transform.

#### shareparts
A network for 3D flow, but sharing some weights to emphasize homogeneous direction.


### Loss_fn
---------------------------
Options under **train_config**.

#### physicswithdata
Insert observation into the prediction before taking physics loss.

The full dataset is $\boldsymbol{u}$, which contains $u,v$ and $p$ (and $w$ if 3D).
The predicted dataset is $\boldsymbol{y}$.
Sensors are placed on $\boldsymbol{x}_o$, and coordinated with no sensors are $\boldsymbol{x}_h$.
The network is represented by $F$.
Residual of the Navier-Stokes equations is calculated with operator $R$.
If a flow field, $\hat{\boldsymbol{u}}$, is a solution to the NS equations, $R(\hat{\boldsymbol{u}}) = 0$.
Steps to calculate the loss is 

1. predict $\boldsymbol{y} = F(\boldsymbol{u})$,
1. set $\boldsymbol{y}\_{\boldsymbol{x}\_o} = \boldsymbol{u}\_{\boldsymbol{x}\_o}$,
    $\tilde{\boldsymbol{y}}\_{\boldsymbol{x}\_h} = \boldsymbol{y}\_{\boldsymbol{x}\_h}$
1. $Loss = \lambda\_p R(\tilde{\boldsymbol{y}}) + \lambda\_s MSE(\boldsymbol{y}\_{\boldsymbol{x}\_o},\boldsymbol{u}\_{\boldsymbol{x}\_o})$.


- `weight_physics` (float:1.0)<br>
    $\lambda\_p$.
- `weight_sensors` (float:0.0)<br>
    $\lambda\_s$.


#### physicsnoreplace 

The full dataset is $\boldsymbol{u}$, which contains $u,v$ and $p$ (and $w$ if 3D).
The predicted dataset is $\boldsymbol{y}$.
Sensors are placed on $\boldsymbol{x}_o$, and coordinated with no sensors are $\boldsymbol{x}_h$.
The network is represented by $F$.
Residual of the Navier-Stokes equations is calculated with operator $R$.
If a flow field, $\hat{\boldsymbol{u}}$, is a solution to the NS equations, $R(\hat{\boldsymbol{u}}) = 0$.

$Loss = \lambda\_p R(\boldsymbol{y}) + \lambda_s MSE(\boldsymbol{y}\_{\boldsymbol{x}\_o},\boldsymbol{u}\_{\boldsymbol{x}\_o})$.


- `weight_physics` (float:0.1)<br>
    $\lambda\_p$.
- `weight_sensors` (float:0.9)<br>
    $\lambda\_s$.
