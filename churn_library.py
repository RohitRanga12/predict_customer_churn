# library doc string
"""
This module is a library of functions to find customers who are likely to churn
Author: Rohit
Date: 29 Jan 2023
"""

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib 
import numpy as np

def import_data(pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''	
        df = pd.read_csv(pth, index_col=0)
        return df


def perform_eda(df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        store_path = "./images/eda/"
        plt.figure(figsize=(20,10)) 
        df['Churn'].hist()
        plt.savefig(store_path+'churn_histogram.png')

        plt.figure(figsize=(20,10)) 
        df['Customer_Age'].hist()
        plt.savefig(store_path+'customer_age_histogram.png')

        plt.figure(figsize=(20,10)) 
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(store_path+'marital_status_bar_chart.png')

        plt.figure(figsize=(20,10)) 
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(store_path+'total_transactions_histogram.png')

        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig(store_path+'correlation_heatmap.png')


def encoder_helper(df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        for category in category_lst:
                cat_lst = []
                cat_groups = df.groupby(category).mean()['Churn']

                for val in df[category]:
                        cat_lst.append(cat_groups.loc[val])

                df[category+'_Churn'] = cat_lst  
        return df


def perform_feature_engineering(df, response):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                'Income_Category_Churn', 'Card_Category_Churn']
        X = pd.DataFrame()
        y = df['Churn']        
        X[keep_cols] = df[keep_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
        return X_train, X_test, y_train, y_test




def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        store_path = './images/results/'
        rf_test_report = classification_report(y_test, y_test_preds_rf, output_dict=True)
        rf_test_report_df = pd.DataFrame(rf_test_report).transpose()
        rf_test_report_df.to_html(store_path+'rf_test_report_df.html')

        rf_train_report = classification_report(y_train, y_train_preds_rf, output_dict=True)
        rf_train_report_df = pd.DataFrame(rf_train_report).transpose()
        rf_train_report_df.to_html(store_path+'rf_train_report_df.html')

        lr_test_report = classification_report(y_test, y_test_preds_lr, output_dict=True)
        lr_test_report_df = pd.DataFrame(lr_test_report).transpose()
        lr_test_report_df.to_html(store_path+'lr_test_report_df.html')

        lr_train_report = classification_report(y_train, y_train_preds_lr, output_dict=True)
        lr_train_report_df = pd.DataFrame(lr_train_report).transpose()
        lr_train_report_df.to_html(store_path+'lr_train_report_df.html')


def feature_importance_plot(model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        # Save the plot to path
        plt.savefig(output_pth+'feature_importance_plot.png')


def train_models(X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        # Train the models 
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        param_grid = { 
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth' : [4,5,100],
                'criterion' :['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)

        # Store the model results 
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)
        classification_report_image(
                y_train,
                y_test,
                y_train_preds_lr,
                y_train_preds_rf,
                y_test_preds_lr,
                y_test_preds_rf
        )

        # Save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

if __name__=='__main__':
        path = r"./data/bank_data.csv"
        cat_columns = [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'                
        ]
        feature_importance_output_path = "./images/results/"
        df = import_data(path)
        perform_eda(df)
        df = encoder_helper(df, cat_columns, '')
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, '')
        train_models(X_train, X_test, y_train, y_test)
        rfc_model = joblib.load('./models/rfc_model.pkl')
        feature_importance_plot(rfc_model, X_train, feature_importance_output_path)

