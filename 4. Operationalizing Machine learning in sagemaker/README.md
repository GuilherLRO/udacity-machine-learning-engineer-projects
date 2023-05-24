# Operationalizing-an-AWS-ML-Project

## Step 1: Training and deployment on Sagemaker

To run the code in `train_and_deploy-solution.ipynb`, I created a notebook instance of type `ml.t3.medium` because, at the time of this submission, it costs only 0.05 USD per hour and should provide enough computational power for our purposes.

### Notebook Instance
![NotebookInstance](./1.%20Notebook%20Instance.png)

### S3 bucket
![S3Bucket](./2.%20S3%20Bucket%20Train%20Test%20Data.png)

### Tunning Jobs
![TuningJobs](./3.%20Tuning%20Jobs.png)

### Deployed Endpoint
![DeployedEndpoint](./4.%20Deployed%20Endpoint.png)

### EC2 Training

For training in EC2 the instance chosen aws m5.xlarge that is powerful enough for the training but still fairly cheap costing 0.23 USD per hour, but I even used a spot instance for it, whar made the cost considerebly lower. The historical average discount for m5.large is 31% for spot instances.

In the EC2 instance, the training was made using the code in 'hpo.py' that using modules that are specific to Sagemaker does all parametrization via command line.

![EC2Training](./5.%20EC2%20Training.png)

### Lambda Function

To invoke the deployed end point a lanbda function was creatd, but it is not allowed to do this kind of task by defalt. To solve this, a new polocy was atached in the AWS IAM to the AWS function role.
![IAMLambdaFunction](./6.%20IAM%20Lambda%20Function.png)

### Lambda Function Execution Test

The lambda function was successfully executed using the atached "SagemakerFullAcess" policy. It should be noted that this can expose your system to security vulnerabilities. So in a real world application, the engineer shoul always be aware of the usage of functions ang give only the most necessary privileges to any function so he or she can ensure a secure operation.

![LambdaFunctionExecution](./7.%20Lambda%20Function%20Execution.png)

### Lambda Function Concurrency

For being able to deal with multiple request at the same time, concurency was configured to the lambda function.
![LambdaFunctionConcurrency2](./8.%20Lambda%20Function%20Concurrency.png)

### Endpoint Scaling Configuration

For being able to deal with high trafic, I also configured varianting automatic scaling, so acording to the parameters shown, the number of dedicted instances can go up to 3.
![EndpointScalingConfig](./9.%20Endpoint%20Scaling%20Config.png)