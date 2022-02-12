# Fashion MNIST ML Pipeline

A machine learning pipeline for training a convolutional neural network on the 
[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## Development Environment Setup

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Create an environment `conda create -n fashmnist python=3.9 -y`
- Install [FastAI](https://github.com/fastai/fastai) with its dependencies
    `conda install -c fastchan fastai`
- Install this package in editable mode `pip install -e .`

## Running the Pipeline

The ML pipeline has been created using [DVC](https://dvc.org/)

### DVC Quick Reference

Commands to run from the terminal:

- `dvc repro`: Run all stages which will produce results not currently in your workspace 
- `dvc repro -f`: Run all stages irregardless of the state of your current workspace
- `dvc repro {STAGE-NAME}`: Run only the stages up to and including {STAGE-NAME}
- `dvc dag`: See a visualisation of the ML pipeline
- `dvc metrics show`: See your workspace results
- `dvc metrics diff`: Compare your workspace results to HEAD results
- `dvc plots show`: Generate plots as HTML file, showing workspace results
- `dvc plots diff`: Generate plots as HTML file, comparing your current results against the HEAD results

### Current Pipeline

```
+---------------------+                                
| download_train_data |                                
+---------------------+                                
            *                                          
            *                                          
            *                                          
    +-------------+            +--------------------+  
    | train_model |            | download_test_data |  
    +-------------+            +--------------------+  
                  ***            ***                   
                     **        **                      
                       **    **                        
                     +----------+                      
                     | evaluate |                      
                     +----------+ 
```

## Deployment Suggestions

Depending on your non-functional requirements, there are a variety of ways you could choose to deploy the model trained 
by this ML pipeline.

### Build Your own RESTful Web API

If you want an always available, small load endpoint; you could build a RESTful web service.

This web-service could receive a HTTP POST request.
Inside the body of this message the image could be encoded as a base64 string.
The web service could then respond with a JSON payload containing the predicted class inside.

For this I recommend using [FastAPI](https://fastapi.tiangolo.com/).

This web service could then be deployed using uvicorn and install it on an EC2 server or an ECS cluster.

### AWS SageMaker

You can import your model into [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/index.html)
which has a suite of deployment options for registered models:
- [Real-time inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)
- [Serverless inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [Batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
- [Asynchronous inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)
- 