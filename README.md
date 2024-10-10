## Getting started
Install python 3.11
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3-pip
sudo apt install python3.11-dev
sudo apt install python3.11-venv
```

Create virtual env
```bash
python3.11 -m venv venv
source venv/bin/activate
pip3 install # your desired version of torch>=2.0
pip3 install -r requirements.txt
mypy --install-types
```

Run example bert training
```bash
mypy app_bert.py # check if code type checks

EXP_NAME="${experiment_name}" RESUME="${resume_flag}" python3 app_bert.py
```

View the logs
```bash
tensorboard --logdir runs
```

## Assumptions
1. The goal of the `simple_lib` library is to help producing trained model checkpoints
2. An application using `simple_lib` is responsible for building the model with appropriate signature and passing it to the `simple_lib` training function 

## Code structure
 - simple_lib:
    - \_\_init\_\_ - paths to resources and global training options 
    - data - download datasets from the web, preprocess the data
    - logging - setup stdout logger for instant feedback
    - nn - a library of commonly used modules/models
    - resources - files bundled for ease of use
    - training:
        - train_bert - file with the definition of bert training loop
        - train_*xyz* - file with the definition of *xyz* training loop
 - app_bert - an application using `simple_lib` to ensemble and train a bert model
 - app_*xyz* - an application using `simple_lib` to ensemble and train a *xyz* model