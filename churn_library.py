# library doc string
"""
This module is a library of functions to find customers who are likely to churn
Author: Rohit
Date: 29 Jan 2023
"""

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    Returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(pth, index_col=0)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    store_path = "./images/eda/"
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig(store_path + 'Churn.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(store_path + 'Customer_Age.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(store_path + 'Marital_Status.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(store_path + 'Total_Trans_Ct.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(store_path + 'heatmap.png')
    plt.close()


def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            dataframe: pandas dataframe with new columns for
    '''
    for category in category_lst:
        cat_lst = []
        cat_groups = dataframe.groupby(category).mean()['Churn']

        for val in dataframe[category]:
            cat_lst.append(cat_groups.loc[val])

        dataframe[category + '_Churn'] = cat_lst
    return dataframe


def perform_feature_engineering(dataframe):
    '''
    input:
            dataframe: pandas dataframe

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    x_values = pd.DataFrame()
    y_values = dataframe['Churn']
    x_values[keep_cols] = dataframe[keep_cols]
    x_training, x_testing, y_training, y_testing = train_test_split(
        x_values, y_values, test_size=0.3, random_state=42)
    return x_training, x_testing, y_training, y_testing


def classification_report_image(y_training,
                                y_testing,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_training: training response values
            y_testing:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    store_path = './images/results/'
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_testing, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(store_path + 'rf_test.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 0.6, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_training, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(store_path + 'rf_train.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_testing, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(store_path + 'lr_test.png')
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 0.6, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_training, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(store_path + 'lr_train.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save the plot to path
    plt.savefig(output_pth + 'feature_importance_plot.png')

    # Close the plot
    plt.close()


def train_models(x_training, x_testing, y_training, y_testing):
    '''
    train, store model results: images + scores, and store models
    input:
            x_training: X training data
            x_testing: X testing data
            y_training: y training data
            y_testing: y testing data
    output:
            None
    '''
    # Train the models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_training, y_training)
    lrc.fit(x_training, y_training)

    # Store the model results
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_training)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_testing)
    y_train_preds_lr = lrc.predict(x_training)
    y_test_preds_lr = lrc.predict(x_testing)
    classification_report_image(
        y_training,
        y_testing,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    PATH = r"./data/bank_data.csv"
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    FEATURE_IMPORTANCE_OUTPUT_PATH = "./images/results/"
    df = import_data(PATH)
    perform_eda(df)
    df = encoder_helper(df, cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
    rfc_model = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(rfc_model, X_train, FEATURE_IMPORTANCE_OUTPUT_PATH)
