# Image Classification using AWS SageMaker
## This assignment is a part of AWS Machine Learning Engineer Nanodegree Program.

Using AWS Sagemaker to train a pretrained model Resnet50 to perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning
The following tasks are performed.
- Using a Resnet50 pretrained model from pytorch library(https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html)
- Fine-tune the model with hyperparameter tuning
- Using the Sagemaker profiling, debugger
- Deploy the model and perform inference

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 
You can use this link to get the starter [files](https://github.com/udacity/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter)
## Dataset
Udacity's Dog Breed Classification Data set is used.
The dataset can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).


### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Files Used

- `train_and_deploy.ipynb` it contains all step to upload data to s3, fine tunning, get best model hyperparameters, train the bestperformance model and test it , and make sagemaker profiling and debugger, and finally deploy the model and make an inference.

- `hpo.py` This is the python script using to train and test all models in tuning hyperparameters step.

- `train.py` This is the python script using to train the best performance model and test it.

- `inference` This script we use it to deploy the model on AWS and make prediction.


## Hyperparameter Tuning
- The Resnet50 used to learn the data because it is trained on a lot of data and it's concolution can get the general feature 
- One fully connected layer is used on top of resnet50 to predict 133 number of dog breed
- Batch- size, Epochs, and Learning rate are used to search for the best model in  the tunning hyperparameters step
     - batch size : [32,64,128,256,512]
     - epochs : (6,8)
     - lr : (0.001,0.1)
 ![Completed Training Jobs](images/hyperparametesjobs.png "completed training jobs")
 ![Logs metrics during the training process](images/training-job-log.png "Logs metrics during the training process")
 ![All Jobs](images/hyperparametes.png "All hyperparameters jobs")
 ![Best Training Job Hyperparameters](images/Bestjob.png)
 

## Debugging and Profiling
### Debugger Output
The Graphical representation of the Cross Entropy Loss.
![Cross Entropy Loss](images/tensor_plot.png "Cross Entropy Loss")
### Profiler Output
The profiler report can be found [here](ProfilerReport/profiler-report.html).

## Model Deployment
- Model was deployed to a "ml.m5.large" instance type and "endpoint_inference.py" script is used to setup and deploy our working endpoint.
- For testing purposes ,one test images are stored in the "images" folder.
- image are fed to the endpoint for inference.
![image](images/Affenpinscher_00078.jpg "image")
### Inference Response
![Inference Response](images/response.png "Inference Response")
![prediction](images/Affenpinscher_00078.png "prediction")

### Model Endpoint
![End Point Deployment](images/endpoint.jpg "End Point")

### Invoke Endpoint by Lambda Function
![Invoke Result](images/lambda-result.png "invoking result")

### Securing Lambda function and endpoint connection with IAM role
- for security manner, we add only the specific role to perform this task not Full Access to Sagemaker to limit the access for lambda and limit any security vulnerabilities.
![Securing connection](images/IAM-role-for-lambda.png "IAM role")

## High Troughput and Low Latency Handling (concurrency and auto-scaling)
- For concurrency I choose the Provisioned concurrency to make instances always on without requiring a wait for start-up time and that achieve low latency in high traffic and I choose to make 3 instance for concurrency and that cost 4.19$ in addition to pricing for duration and requests. 

![Lambda concurrency](images/Lambda-concurrency.png "Lambda Concurrency")
- For Auto-Scaling to deal with high-requested gets from Lambda Function I use 3 maximum instance count for auto scaling and in Scale in cool down I will configure to start a new instance after 10 second if I get more than 15 requests in the same time and if I get less than 15 request in the same time for 1 minutes I close the additional instance that uses for deal with high throughput
![Endpoint Auto-scaling](images/auto-scaling-endpoint.png "End Point auto-scaling")