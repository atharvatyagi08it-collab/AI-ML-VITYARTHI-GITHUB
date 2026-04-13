# AI-ML-VITYARTHI-GITHUB
This project is a Student Performance Predictor that uses the concept of Linear Regression to estimate a student’s marks based on key factors such as study hours, sleep hours, and attendance percentage. It works by first creating a dataset and training a model to learn the relationship between these inputs and the corresponding marks.  

ATHARVA TYAGI-AIML-project
Student Performance Predictor
A simple machine learning project that predicts student marks using Linear Regression.

Project Idea
Predict marks based on:

Study hours ( amount of student spent while studying )
Sleep time ( amount of time student invested during rest )
Attendance ( total number of class in which student was present )
Concepts Used
Linear Regression
Feature-based prediction
Train-test split
Model evaluation (MAE, R2 score)
Tech Stack
Python
Pandas
Scikit-learn
Input and Output
Input: Hours studied, sleep time, attendance percentage
Output: Predicted marks out of 100
Files
student_performance_predictor.py - Main script for training and prediction
Installation
Install dependencies:

pip install pandas scikit-learn
Run the Project
python student_performance_predictor.py
How It Works
Builds a sample containing student's dataset
Trains a Linear Regression model
Evaluates model performance using MAE and R2 score
Takes user input for new student data
Predicts and displays marks
Example Input
Study hours per day: 6
Sleep hours per day: 7
Attendance percentage: 85
Example Output
Predicted Marks: around 75-85 (depends on trained model)
Learning Outcomes
This project helps you practice:

Basic machine learning workflow
Data handling with Pandas
Regression modeling with Scikit-learn
User input validation in Python

