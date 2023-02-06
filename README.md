# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- This is a project to implement best coding practices. 
- Goal: Identify credit card customers that are most likely to churn, by implementing the engineering and software best practices

## Files and data description
Overview of the files and data present in the root directory. 

=> data/
- contains csv with the data for customer churn

=> images/eda
- contains images of EDA performed on the different columns of the data

=> images/results
- contains images of feature importance for Random Forest classifier model
- contains classification reports of Logistic Regression and Random Forest models on train and test datasets

=> models/
- contains the trained (best performing) Logistic Regression and Random Forest models

=> logs/
- contains the log file that is created when running unit tests 

## Running Files
How do you run your files? What should happen when you run your files?

- cd into the project root directory

- activate the conda environment that contains all required packages for running the project, by running the below command 
    
    <code>conda create --name <env> --file requirements_py3.8.txt</code>

- run the following commands:

    <code>python churn_library.py</code>

        - This will generate the .png files inside images/eda and images/results folders
        - This will also generate the .pkl model files inside models/ folder

    <code>python churn_script_logging_and_tests.py</code>

        - This will generate the .log file inside logs/ folder

- in order to test the pep8 styling, run the below commands

    <code>pylint churn_library.py</code>
    
    <code>pylint churn_script_logging_and_tests.py</code>




