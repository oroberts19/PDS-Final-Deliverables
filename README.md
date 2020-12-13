# Practical Data Science Project- Dealer's United

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project aims to:
* Understand what makes an advertisement clickable to predict future advertisement performance using a random forest and mixed input neural network
* Explore features that predict the price point of a vehicle and use to predict future prices using a baseline model, random forest and linear regression    

## Technologies
* Python 3.5–3.8 using Anaconda (downloadable here: https://www.anaconda.com/products/individual) or if using a Python interpreter, `pip install` all packages and skip conda environment
* TensorFlow 2.3.1

## Setup  
Please ssh username@10.10.11.64 and navigate to /home/oroberts/DealersUtd to use as working directory for proof of concept. All data and both .py files are there. Run `python3 pricing_model.py` or `python3 engagement_rate_model.py` for respective models.  

Please create a conda environment replacing myenv with your choice of name  
`conda create --name myenv`

Activate environment    
`conda activate myenv`  

Install the following libraries:  
`conda install -c anaconda pandas`    
`conda install -c anaconda matplotlib`    
`conda install -c anaconda numpy`   
`conda install -c anaconda urllib3`    
`conda install -c anaconda pillow`    
`conda install -c anaconda scikit-learn`    
`pip install tensorflow==2.3.1`  
`pip install textblob`  
`pip install nltk`
`pip install rake-nltk`    
`pip install --upgrade category_encoders`    
`pip install shap`    
`pip install seaborn`  
`pip install bs4`  
`pip install lxml`  
`conda install -c conda-forge tqdm`  

In total the following files will be used:  
data_09092020/vehicles_20200909  
all_features_df.csv  
filtered_df.csv  
id_images.csv  
jpg_images  
new_pricing.csv  
performance.csv  
performance_description.csv  
perform_features.csv  
sample_pricing.csv  
saved_model.pb  

## Contact
For any questions, please contact Olivia Roberts (olivia.roberts19@ncf.edu), Jeeda AbuKhader (jaida.abukhader15@ncf.edu), or Simona Rahi (simona.rahi15@ncf.edu)
