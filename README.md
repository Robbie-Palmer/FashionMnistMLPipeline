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