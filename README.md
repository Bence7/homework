# Multi-Label Classification with Neural Network on Iris data set
Welcome to the `homework` project. The goal of this project is to make the deployment of ML models straightforward, efficient, and maintainable. The project has been set up to handle each aspect of the ML pipeline from data processing, training models, and inferencing on new data.

### MLFlow on Windows

To run MLFlow, type `mlflow ui` in your terminal and press enter. If it doesn't work, you may also try `python -m mlflow ui`  This will start the MLFlow tracking UI, typically running on your localhost at port 5000. You can then access the tracking UI by opening your web browser and navigating to `http://localhost:5000`.


## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── iris_inference_data.csv
│   └── iris_train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.

## Data:
For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -dit training_image /bin/bash
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/<model_name>.pth ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pth` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pth --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -dit inference_image /bin/bash  
```
After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.
