import os
import logging
from churn_library import import_data, perform_eda

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		df = import_data("./data/bank_data.csv")
		df = perform_eda(df)
		assert len(os.listdir('./images/eda')) == 5
		logging.info("EDA performed successfully")
	except AssertionError:
		logging.error("The EDA results have not been properly captured and stored")

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	We test that the number of columns ending with _Churn should be 5
	'''
	try:
		df = import_data("./data/bank_data.csv")
		df = perform_eda(df)
		logging.info()
	except AssertionError: 
		logging.error("")



def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








