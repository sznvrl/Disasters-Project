# Disasters-Project


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Detail Information for Running](#running)
4. [Results](#results)

## Installation <a name="installation"></a>

Python 3.0 used in this analyze. Matplotlib, seaborn, sqlalchemy, nltk, re, numpy, plotly, string, pickle,plotly were used.


## Project Motivation<a name="motivation"></a>
In this project,  messages were analyzed. Our goal is to assign disaster messages to categories in the most correct way.


## File Descriptions <a name="files"></a>
* **disaster_categories.csv**: CSV file; categories dataset for all messages
* **disaster_messages.csv**: CSV file; sample messages dataset
* **ETL Pipeline Preparation.ipynb**: cleaning dataset; merge two dataset (messages and categories), check duplicates etc. 
* **ML Pipeline Preparation.ipynb**: apply ML models for classifying messages and calculate evaluation metrics 
* **process_data.py**: give data as a input, cleaned data and write to SQLite
* **train_classifier.py**: takes cleaned data from SQLite and apply ML models for classifying messages, print ML scores
* **run.py**: visualize results and create app



## Detail Information for Running <a name="running"></a>

* **Step1**: Check direction for all processes. Firstly run process_data.py and give inputs (disaster_categories.csv, disaster_messages.csv and write cleaned data in DisasterResponse database. In this study, process_data.py ,disaster_messages.csv and disaster_categories.csv files in "data" folder.

***cd data**
***python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db**

* **Step2**: 

python train_classifier.py data/DisasterResponse.db classifier.pkl




## Results<a name="results"></a>
According to result, the sales prices are graphed.
