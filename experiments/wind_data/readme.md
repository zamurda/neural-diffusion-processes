
# Wind turbine modelling experiment

This directory contains an experiment aimed at modelling power curves for single/multiple turbines.

## Training Instructions

### Single Turbine

This model uses the standard Neural Diffusion Process model. To train the model, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using poetry.
3. Configure your model by editing the `config.toml` file
4. Run the training script: `python train.py --config ./config.toml`
5. Once the training is complete, the trained model will be saved in the `./trained_models` directory.
6. Run ain with `aim up` to check the model metrics while it is training.


### Multiple Turbines

This model uses the new multi-channel Neural Diffusion Process model. To train the model, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using poetry.
3. Configure your model by editing the `config_multi.toml` file
4. Run the training script: `python train_multi.py --config ./config_multi.toml`
5. Once the training is complete, the trained model will be saved in the `./trained_models` directory.
6. Run ain with `aim up` to check the model metrics while it is training.


### Using a GPU

If you have a GPU available, simply add `--metal` to the command when running the training script.

