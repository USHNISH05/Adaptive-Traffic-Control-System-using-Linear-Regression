# Adaptive-Traffic-Control-System-using-Linear-Regression

This project aims to predict traffic signal green times using the Linear Regressor models like gradient boosting, polynomial, decision tree, random forest from the scikit-learn library. The green times are predicted based on both the present and upcoming traffic conditions in the East-West (EW) and North-South (NS) directions based on various features.

Dataset
The dataset used for this project is stored in traffic_signal_data.csv and includes the following features:

cycle_length: Length of the traffic signal cycle in seconds.
lane1_present_num_vehicles: Number of vehicles currently in lane 1.
lane2_present_num_vehicles: Number of vehicles currently in lane 2.
lane3_present_num_vehicles: Number of vehicles currently in lane 3.
lane4_present_num_vehicles: Number of vehicles currently in lane 4.
lane1_upcoming_num_vehicles: Number of vehicles expected in lane 1.
lane2_upcoming_num_vehicles: Number of vehicles expected in lane 2.
lane3_upcoming_num_vehicles: Number of vehicles expected in lane 3.
lane4_upcoming_num_vehicles: Number of vehicles expected in lane 4.
ew_present_waiting_times: Current waiting times for East-West direction in seconds.
time_of_day: Time of the day in hours (0-23).
flow_ratios: Flow ratios of the vehicles.
The target variables are:

ew_present_green_time: Present green time for East-West direction in seconds.
ns_present_green_time: Present green time for North-South direction in seconds.
ew_upcoming_green_time: Upcoming green time for East-West direction in seconds.
ns_upcoming_green_time: Upcoming green time for North-South direction in seconds.

Requirements
Python 3.x
pandas
scikit-learn

Model Training and Prediction
Data Loading: The data is read from a CSV file.
Feature and Target Definition: Features include various traffic-related parameters, and the targets are the present and upcoming green times for EW and NS directions.
Data Splitting: The data is split into training and testing sets.
Model Definition and Training: Gradient Boosting Regressor models are defined and trained for each target variable.
Prediction: The trained models predict the green times on the test set.
Error Calculation: Mean Squared Error (MSE) is computed to evaluate model performance.

Results
The models compute the predicted green times for both the present and upcoming traffic conditions in EW and NS directions. If the predicted upcoming green time is greater than the present green time, an average of the two is taken as the final green time. Otherwise, the present green time is used.

Summary
The project demonstrates how to use Linear Regressor models models to predict traffic signal green times based on traffic data. It involves data preprocessing, model training, prediction, and evaluation, providing a comprehensive approach to traffic signal timing optimization.







