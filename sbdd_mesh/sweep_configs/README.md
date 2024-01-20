# Use wandb sweep to find hyperparameters

## 1. Define a sweep configuration
- a `.yaml` file for wandb
- a `.sh` file containing default arguments (which are not included in `.yaml`) to run the `sweep.py` file 

## 2. Creat sweep and get id
- run code in command line: `wandb sweep --project $YOUR_PROJECT $PATH_TO_YAML`. It will generate a sweep id. 
- Add sweep id to the `.sh` file

## 3. run an agency
- run code in command line: `./sweep_configs/$PATH_TO_SHELL`