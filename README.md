# Comparison of models for predicting electricity price. 

Predicting the electricity price may look like an easy task, just as the forecasting of assets. In fact, there are a few ways for achieving this goal. The purpose of this project is to check and measure this differents paths to get the prediction of the electricity price. 

In this frontpage we are explaining the main structure of the project, how it is divided and how to go through it. If you are looking for a deeper study of the techniques and methods used, you can check the [memory attached](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Comparison_of_ML_models.pdf) in this repository where everything has been detailed.

The project is updated everytime it is run, so it is always up to date. It is important to say that some of the notebooks take quite a lot of time because of the huge amount of information it has to handle. In order to be able to study the code and parallelize tasks, we have created a csv file at the end of each notebook for the next step. For example, at the end of the recollection of the data we export a csv file with all the information for the exploratory analysis, so if you want to check if the code is running perfectly, you can run both notebooks at the same time, because one will load a new csv but the other one is looking at the old csv. 

Using default csv files is only recommended if you want to check the results from the document, which are the results for the 5th of September of 2020, or if you want just to check if the code runs without error. If you would like to use the project itself, the results from the memory may vary and the only thing to do is just running the whole project following the index.

## Structure of the project and how to run it.
### Index
**0. [Preparing the environment](#section_0).**

**1. [Installing modules](#section_1).** Link to the [folder](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/tree/master/Modules).

**2. [Recollecting data](#section_2).** Link to the [folder](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/tree/master/Getting_data).

**3. [Exploratory analysis and data transformation](#section_3).** Link to the [folder](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/tree/master/Exploring_data).

**4. [Models](#section_4).** Link to the [folder](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/tree/master/Models).

**5. [Visualization](#section_5).** Link to the [folder](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/tree/master/Visualization).

<a id='section_0'></a>
### 0. Preparing the environment.

First of all, we have to create the appropiate environment. You must follow the [instructions](https://docs.google.com/document/d/1cwYA04H2VXTalkR78nYHDvJmm1QzhEZjSC_4WvjB1Vw/edit) that were given to us at the beginning of the master. 

Once we have linux and Anaconda installed, we can start with the project itself. For initialazing jupyter, the following command should be used: "**jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10**". This is because for getting the information of the Temperature, several attemps must be sent, so in order to ensure a correct behavour of the connection, the rate limit must be bigger than the default one. 

<a id='section_1'></a>
### 1. Installing modules. [Notebook](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Modules/Installing_modules.ipynb)

After having prepared the environment, the next step is to install all the libraries that we are using in Python. The only thing that is needed here is running the notebook and all the libraries will be installed automatically.

<a id='section_2'></a>
### 2. Recollecting data. [Notebook](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Getting_data/Getting_Data.ipynb)

Once we have the libraries installed, the next notebook can be run. Here we are downloading all the information that we are using for the models and merging it into one single dataframe. It takes approximately 1:15 hours to be completed, so other notebooks can be checked while this one is running if you are just checking the code. **Try to run this notebook only for checking purposes unless you have enough time. **

<a id='section_3'></a>
### 3. Exploratory analysis and data transformation. [Notebook](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Exploring_data/Exploratory_analysis.ipynb)

After getting all the information, the next notebook allows us to study the information and get a deeper insight. It takes around 20 minutes to complete this notebook. With this code an exploration of the data will be displayed where all the variables can be analized and the correlations between them.

<a id='section_4'></a>
### 4. Models. [Gradient Boosting regressor](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Models/Regressor.ipynb), [Time Series models](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Models/Time_series.ipynb) and [Neural Networks](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Models/Neural_Network.ipynb)

The last part of the investigation are the models. Three kind models have been fitted to the data:

- XGBoost: The first model is based on the XGBoost algorithm. Two tests have been made with this algorithm. No interaction is needed, just run all the cells. It takes approximately 25 minutes to complete. 
- Time series models: Two algorithms have been used here, Facebook prophet and SARIMAX. Both of them are in same notebook, code just need to be run, with an execution time of around 30 minutes for the whole code. 
- Neural Networks: The last models are the neural networks. The data has been tested with LSTM neural networks and DNN. Results can be seen by running the code. This is the models notebook that takes the longest to execute, so **try not to run the LSTM code if you don't have enough time** (it takes around 50 minutes for the LSTM algorithm to be trained and around 35-40 minutes for both DNN algorithms). 

<a id='section_5'></a>
### 5. Visualization. [Notebook](https://github.com/JoelDela/TFM-Master_Data_Science_Kschool/blob/master/Visualization/Visualization.ipynb)

Finally, we can display the visualization as it is indicated in the memory. In short, the only thing needed is to run the notebook and go to the link that is displayed in the results (usually http://192.168.1.43:8501).
