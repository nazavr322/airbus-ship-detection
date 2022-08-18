# Project overview
A repository for the Winstars Technology test task - **[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview).**    
**NOTE:** attached to this project jupyter notebook with all the EDA and visualizations happened to be too huge for github to render it. If you don't want to clone this repository to only check out notebook content's, you can find the rendered version [here](https://nbviewer.org/github/nazavr322/airbus-ship-detection/blob/main/notebooks/eda_and_visualization.ipynb).
## Project structure
```nohighlight
├── README.md          <- The top-level README for developers using this project.
├── data 
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized model weights, hyperparamets specification
│							
│
├── notebooks          <- Jupyter notebooks.
│	└── eda_and_viz.ipynb    <- Notebook with EDA, visualization and demonstrations of predictions.
│								  
│── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. 
│   └── figures    <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file generated with `pip freeze > requirements.txt` to intall dependencies via pip.
│							
│
├── conda_req.txt      <- The requirements file generated with `conda list --export > conda_req.txt` to install dependencies via conda 
│							 							      
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to process and generate data
│   │	├── balance_data.py       <- Script to balance number of classes in a dataset.						  
│   │   ├── datasets.py           <- File with dataset's definitions.
│   │   ├── functional.py         <- File with utility functions relatet to data processing.						  
│   │	├── prepare_dataset.py    <- Script to clean data and add features.
│   │   └── split_train_val.py    <- Script to perform train/validation split.	
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── features.py    <- File with functions that generate features.
│   │
│   └── models         <- Scripts to define and train models
│	├── functional.py    <- File with definitions of custom metrics and losses.
│	├── models.py        <- File with model definitions.
│	└── train.py         <- Script to train model.
│ 
├── .dvcignore         <- .gitignore analog for DVC.
│
├── .env               <- File with environmental variables used in this project
│
├── .gitignore         <- Files to ignore
│
├── dvc.lock           <- Information about files tracked by DVC.
│
└── dvc.yaml           <- File with definition of DVC data preprocessing pipeline.
```
## Solution description
I finished with a `Dice score = 0.73`, which I think is a pretty good result. Below I will try to give a brief overview of my work and share some thoughts on what worked and what didn't.
I can divide my work on this problem into a 3  most important parts:
1.  **Data preparation**   
	Understanding how data is structured is one of the main aspects of a successful solution. Therefore, the first thing I took on this project was 	EDA and different visualizations. You can check out full code and all the graphs at `notebooks/eda_and_viz.ipynb` or at the rendered version of this notebook [here](https://nbviewer.org/github/nazavr322/airbus-ship-detection/blob/main/notebooks/eda_and_visualization.ipynb), here I'll try to give you the main idea.   
	1. **The first thing to check is a distribution of our data**
			<img src="https://github.com/nazavr322/airbus-ship-detection/blob/main/reports/figures/distribution_of_data.png">
	As you can see, we have a severe imbalance. Almost 78% of images doesn't have ships on it at all, it is not very useful for ship segmentation task :)
	In order to make situation a little bit better, I a performed a series of transforms.	
	2. **Data processing pipeline and DVC**   
	You can check out all the preprocessing scripts in the corresponding files, but you don't need to worry about understanding and reproducing it 		correctly. I created a processing pipeline using DVC, which allows you to generate ready-to-train .csv files and actually train a model using 		only one command (I will explain how to do it in a corresponding [section](#getting-started).
	Since this is a competition, the pipeline is really simple and straightforward:      
	<p align="center">
  		<img src="https://github.com/nazavr322/airbus-ship-detection/blob/main/reports/figures/dvc_pipeline.svg">
	</p>

	This steps will do the following: clean data from duplicates and add `ShipCount` feature;  select more balanced subset of data; split this subset 	  into train and validation datasets; start training on this data with hyperparameters specified in the corresponding .json file (output of this 	  step will be the weights of your trained model).
2. **Model Architecture**    
	To solve this task, i've used an UNet architecture, which is very popular in many segmentation problems, showing very good results with relatively small amount of parameters.
	Firstly, i've tried to use UNet with a pretrained ResNet50 as an encoder part. Unfortunately, due to the computational constraints of my hardware, I could not train a model of this size. But still, I left all the necessary functions to build UNet with such architecture, so if you have enough memory, you can change a few lines of code and experiment with more powerful model.
	So, I ended up with much smaller UNet, with some deviations from original paper (in my model I have BatchNorm layers for example).
3. **Loss Functions, Metrics and Hyperparameters**    
	Here I will explain my choices of certain hyperparameters.
	1. **Loss functions and metrics**    
		I also wrote it in a notebook, but will repeat here. In a lot of segmentation tasks, Binary Cross Entropy is a very good choice, but in this competition we have a huge imbalance between pixels of one image (by eye ships usually cover ~10% of image at best), so, in my opinion,  using **only** BCE will produce inconsistent results.
		
		I know, that we can achieve better convergence if our loss function will be somewhat similar to the metric we use. So, the first thing i've tried was IoU Score as a metric and IoU Loss (simply defined as $1 - IoU$). It already produced some significant result with `IoU score == 0.58` on validation dataset. But loss decreased slowly and I understood that there must be better solution.

		The combination of Dice Score and BCEDiceLoss (defined as $\beta * (1 - Dice) + (1 - \beta)*BCE$) worked perfect for me, even with equal weights ( $\beta = 0.5$ ) to both losses, I achieved `Dice Score = 0.73` on validation data only after 20 epochs of training.

		*Loss functions that I heard about, but did not have time to test: Focal Loss, Huber Loss, Lovasz Loss*
	2. **Hyperparameters**    
		I don't want to stop here for a long, because my hyperparameters where pretty much default ones.
		20 epochs with batch_size of 64, Adam as optimizer with learning_rate = 1e-3, ReduceLROnPlateau scheduler with patience 2 and a factor of 0.33 which monitors validation loss. I used only 2 simple augmentations, RandomFlip and RandomRotate90. I wanted to add more different transforms to make the data more diverse, but I didn't have enough time.

To sum up, I found this results quite decent. My UNet architecture is relatively small and fast, it doen't have any pretrained encoders and works with images of size 256x256 (also i think decreasing size to 128x128 will give good results too). In addition, it was trained for only 20 epochs, which makes training pretty fast (it took me ~1 hour and 20 minutes to train this model in kaggle notebooks with nvidia K80 GPU).

Taking all of this into account i think model copes with this ship segmentation task pretty good. But also we can see that sometimes model try to classify some piece of land or waves as a ship.    
   
Below you can find some examples of mode predictions.    
![6_to_10](https://github.com/nazavr322/airbus-ship-detection/blob/main/reports/figures/6_to_10_ex.png)    
![10_to_15](https://github.com/nazavr322/airbus-ship-detection/blob/main/reports/figures/11_to_15_ex.png)

# Getting started
I am using python version `python 3.10.4` in this project.    
After you cloned a git repo:
- Unzip the [data](https://www.kaggle.com/c/airbus-ship-detection/data) in a `./data/raw/` directory.
- I strongly recommend you to install dependencies using [`conda`](https://docs.conda.io/en/latest/) because it simplifies the process of installing `tensorflow` library. To install all needed dependencies create new conda virtual environment and run `conda install --file conda_req.txt`.    
	If for some reason you can't/don't won't to install packages via `conda`, but you want to use GPU,  you can install dependencies with `pip`. But before that, [verify](https://www.tensorflow.org/install/pip#hardware_requirements) that you have compatible versions of CUDA software.
	When it's done, create new `venv` environment and run     
	`pip install -r requirements.txt`
- Now, to generate processed data, run `dvc repro` command from the project root folder. It will generate all needed .csv files step-by-step and then start the training process. 

After this you can continue experimenting with model hyperparameters or explore jupyter notebook with EDA and visualizations!


