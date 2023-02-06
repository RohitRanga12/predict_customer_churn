# library doc string
"""
This module is to test the functions in churn_library.py script
Author: Rohit
Date: 6 Feb 2023
"""

# import libraries
from os import path
import logging
from churn_library import import_data, perform_eda, encoder_helper, \
perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		dataframe = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert dataframe.shape[0] > 0
		assert dataframe.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	dataframe = import_data("./data/bank_data.csv")
	try:
		perform_eda(dataframe)
		features = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct', 'heatmap']
		for feature in features:
			assert path.isfile('./images/eda/'+feature+'.png')
		logging.info("EDA completed successfully")
	except AssertionError as err:
		logging.error("EDA not completed properly. Please check the function")
		raise err


def test_encoder_helper():
	'''
	test encoder helper
	'''
	dataframe = import_data("./data/bank_data.csv")
	perform_eda(dataframe)
	category_lst = [
		'Gender',
		'Education_Level',
		'Marital_Status',
		'Income_Category',
		'Card_Category'
	]
	try:
		dataframe = encoder_helper(dataframe, category_lst)
		for category in category_lst:
			assert category+'_Churn' in dataframe.columns
		logging.info("EDA encoder was successful in its function")
	except AssertionError as err:
		logging.error("EDA encoder not working properly. Please check the function")
		raise err


def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	dataframe = import_data("./data/bank_data.csv")
	perform_eda(dataframe)
	category_lst = [
			'Gender',
			'Education_Level',
			'Marital_Status',
			'Income_Category',
			'Card_Category'
		]
	dataframe = encoder_helper(dataframe, category_lst)
	try:
		x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)
		assert len(x_train) == len(y_train)
		assert len(x_test) == len(y_test)
		logging.info("Data has been split correctly")
	except AssertionError as err:
		logging.error("Data has not been split correctly")
		raise err


def test_train_models():
	'''
	test train_models
	'''
	dataframe = import_data("./data/bank_data.csv")
	perform_eda(dataframe)
	category_lst = [
			'Gender',
			'Education_Level',
			'Marital_Status',
			'Income_Category',
			'Card_Category'
		]
	dataframe = encoder_helper(dataframe, category_lst)
	x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)
	try:
		train_models(x_train, x_test, y_train, y_test)
		assert path.isfile('./models/logistic_model.pkl')
		assert path.isfile('./models/rfc_model.pkl')
		logging.info("Models have been trained and saved successfully!")
	except AssertionError as err:
		logging.error("Models have not been trained and saved properly.")
		raise err

if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_train_models()
