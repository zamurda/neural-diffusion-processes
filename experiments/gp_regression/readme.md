# GP Regression Experiment

This directory contains an experiment aimed at training a model to predict Gaussian Processes (GPs). 

## Training Instructions

To train the model, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using poetry.
3. Configure your model by editing the `config.toml` file
4. Run the training script: `python main.py --config ./config.toml`
5. Once the training is complete, the trained model will be saved in the `./trained_models` directory.
6. Run ain with `aim up` to check the model metrics while it is training.

## Evaluation Instructions

To evaluate the trained model, follow these steps:

1. In `eval.py`, edit the model name and the training step number at which you would like to load the model.
2. Run the evaluation script with `python eval.py --config ./config.toml`.

Feel free to explore the code and experiment with different configurations to improve the model's predictions.
