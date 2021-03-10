# long-short-equities

## Brief Introduction
The project utilizes the cool Python module Streamlit to create a visual interface to build/analyze a Portfolio management strategy using the RandomForest Regressor to predict the prices for n days ahead. It uses the random grid search for inside cross-validation to optimize the hyper parameters(different for every equity) for a chosen number of equities(stocks) by fetching the data from the nsepy module with the ability to select the timeline along with a backtester to test the strategy created.

## To run the script on your local machine
Pull the project from github and ideally, you should create and activate a new virtual environment before installing the necessary modules and running the script.

Firstly install the modules with the help of requirements.txt file by using the following command:
#### "pip install -r requirements.txt"
To run the script, use the following command:
#### "streamlit run long-short-equity.py"

Disclaimer: The project is only meant for fun and serves as a simple representative strategy+backtester for budding systematic trading enthusiasts, it involves no investment advice.

## Preview of the project
![604931f05f73d371179235](https://user-images.githubusercontent.com/45936015/110696664-5c541000-8211-11eb-85f8-93fad0fb8a23.gif)


