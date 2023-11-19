# digit_recognizer
For this project our goal is to correctly identify digits from handwritten images. I will perform an EDA, feature engineer by reshaping the data, transform the data and train multiple models. I will then create a training script, containerized FastAPI app for model deployment and management. I will use keras with tensorflow and use model tuning and data augmentation. 

The dataset [kaggle digit-recognizer data](https://www.kaggle.com/competitions/digit-recognizer/data) 

# Dataset Description
For the project I used the train.csv from kaggle.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

# EDA
I created a notebook for the [EDA and Baseline](baseline.ipynb). Analyzying the dataset we see the target label of 1 to 10 pretty well distributed. Also the dataset is a row of 784 columns representing the pixels which we need to shape into the 28 by 28 shape. Using numpy reshape we are able to reshape the data into the appropriate digit shape. Each shape is now considered a tensor representation. The notebook ilustrates with examples using the tensor and printing the image. After reshapping I also normalized the numerical values from 0-255 to 0-1. CNN's generally perform better with data normalized between 0 to 1. Before training we need to convert the labels y to categorical target features because we will have 10 classes which will be categorized with our model. 

# Model Training
I created a [baseline](baseline.ipynb) model using keras tensorflow Convolutional Neural Network (CNN). I further augmented the baseline using data augmentation. And further experimented with additional layers, dropout and batch normalization. I was able to improve on the baseline of 98.94 to 99.24. 

# Training Script
I created a training script with the best model parameters. Which can be run with the folling Makefile command.
Note: The requirements.txt install is explained in the reproducibiity section.

```bash
make run_train
```

# Reproducibility
The notebook, training script and web app can all be run after the requirements is installed.
```bash
pip install -r requirements.txt
```

The github repository includes everything needed also with a Makefile.

Note: curl is also needed for the client test 

# Model Deployment

The model has been deployed and containerized with FastAPI. The code [FastAPI code](web_service/app/predict.py)

This can also be tested using the Makefile
```bash
make test_server

# which is
uvicorn app.predict:app --reload --app-dir web_service --port 9696
```

In a new window execute using make or the following curl command. row_data.json is a sample json file with test data. 
```bash
make test_client

# Corresponding curl command
curl -X POST -d @row_data.json -H "Content-Type: application/json" http://localhost:9696/predict
```
NOTE: row_data.json is sample data to test with. The digit is labeled 1 

# Dependency and environment management
For example using pyenv
```bash
pyenv virtualenv 3.9.17 digit_recognizer

pyenv activate digit_recognizer
```
Or using your favorite dependency management system

The project has a requirements.txt and can be used for example
```bash
pip install -r requirements.txt
```


# Containerization
The FastAPI application has been dockerized using this [Dockerfile](web_service/Dockerfile)

This can be tested with the following Makefile commands for builing and running

```bash
make docker_build

# Which is
docker build -t digit_recognizer:v1 -f web_service/Dockerfile .
```

```bash
make docker_run

# Which is
docker run -p 9696:9696 digit_recognizer:v1
```

This can then be tested with the makefile
```bash
make test_client
```

# Cloud deployment
TBD
