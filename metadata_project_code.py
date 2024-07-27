# -*- coding: utf-8 -*-
"""Metadata-Project Code.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18YKloEoo5yqmEq142Tpjueqq_j9SUiYD

**Project:** Automated Metadata Management and Data Profiling System for Medical Insurance Cost Analysis.

**Aim:**

Developing a system that automatically manages information about medical insurance costs. This system will use details about the data to make sure it's accurate and organized well. It will also help us understand what's in the data and how it's structured.

**Step 1:** Data Loading, Cleaning, and Profiling with Metadata Management Tools
"""

!pip uninstall -y pandas-profiling
!pip uninstall -y pydantic
!pip install pandas-profiling==3.6.0
!pip install great_expectations
!pip install dataprofiler
!pip install pydantic-settings

import pandas as pd
import pandas_profiling
import great_expectations as ge
from dataprofiler import Profiler, Data

df = pd.read_csv('/content/insurance.csv')

print(df.head())

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

df.fillna(df.mean(), inplace=True)

"""Data Profiling"""

import pandas_profiling

# Generate the profiling report
profile = pandas_profiling.ProfileReport(df, title='Insurance Dataset Profiling Report')
profile.to_notebook_iframe()

profile.to_file("pandas_profiling_report.html")

"""Great Expectations"""

import great_expectations as ge

# Convert the DataFrame to a Great Expectations dataset
ge_df = ge.from_pandas(df)

# Add expectations (example)
ge_df.expect_table_row_count_to_be_between(min_value=1000, max_value=1500)
ge_df.expect_column_values_to_not_be_null('age')
ge_df.expect_column_values_to_be_in_set('sex', ['male', 'female'])

# Validate
validation_result = ge_df.validate()
print(validation_result)

import json
from pathlib import Path

# Sample JSON data with corrected boolean values
json_output = {
    "success": True,
    "results": [
        {
            "success": True,
            "expectation_config": {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {
                    "min_value": 1000,
                    "max_value": 1500,
                    "result_format": "BASIC"
                },
                "meta": {}
            },
            "result": {
                "observed_value": 1338
            },
            "meta": {},
            "exception_info": {
                "raised_exception": False,
                "exception_message": None,
                "exception_traceback": None
            }
        },
        {
            "success": True,
            "expectation_config": {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {
                    "column": "age",
                    "result_format": "BASIC"
                },
                "meta": {}
            },
            "result": {
                "element_count": 1338,
                "unexpected_count": 0,
                "unexpected_percent": 0.0,
                "unexpected_percent_total": 0.0,
                "partial_unexpected_list": []
            },
            "meta": {},
            "exception_info": {
                "raised_exception": False,
                "exception_message": None,
                "exception_traceback": None
            }
        },
        {
            "success": True,
            "expectation_config": {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {
                    "column": "sex",
                    "value_set": [
                        "male",
                        "female"
                    ],
                    "result_format": "BASIC"
                },
                "meta": {}
            },
            "result": {
                "element_count": 1338,
                "missing_count": 0,
                "missing_percent": 0.0,
                "unexpected_count": 0,
                "unexpected_percent": 0.0,
                "unexpected_percent_total": 0.0,
                "unexpected_percent_nonmissing": 0.0,
                "partial_unexpected_list": []
            },
            "meta": {},
            "exception_info": {
                "raised_exception": False,
                "exception_message": None,
                "exception_traceback": None
            }
        }
    ],
    "evaluation_parameters": {},
    "statistics": {
        "evaluated_expectations": 3,
        "successful_expectations": 3,
        "unsuccessful_expectations": 0,
        "success_percent": 100.0
    },
    "meta": {
        "great_expectations_version": "0.18.19",
        "expectation_suite_name": "default",
        "run_id": {
            "run_name": None,
            "run_time": "2024-07-24T11:53:28.194721+00:00"
        },
        "batch_kwargs": {
            "ge_batch_id": "572664b2-49b3-11ef-9705-0242ac1c000c"
        },
        "batch_markers": {},
        "batch_parameters": {},
        "validation_time": "20240724T115328.194612Z",
        "expectation_suite_meta": {
            "great_expectations_version": "0.18.19"
        }
    }
}

# Convert JSON to HTML
def json_to_html(data):
    html_content = "<html><body>"
    html_content += "<h1>Great Expectations Validation Report</h1>"
    html_content += "<h2>Statistics</h2>"
    html_content += f"<p>Evaluated Expectations: {data['statistics']['evaluated_expectations']}</p>"
    html_content += f"<p>Successful Expectations: {data['statistics']['successful_expectations']}</p>"
    html_content += f"<p>Unsuccessful Expectations: {data['statistics']['unsuccessful_expectations']}</p>"
    html_content += f"<p>Success Percent: {data['statistics']['success_percent']}%</p>"

    html_content += "<h2>Results</h2>"
    for result in data['results']:
        html_content += "<div>"
        html_content += f"<h3>Expectation Type: {result['expectation_config']['expectation_type']}</h3>"
        html_content += f"<p>Success: {result['success']}</p>"
        html_content += f"<p>Result: {result['result']}</p>"
        html_content += "</div><br>"

    html_content += "</body></html>"
    return html_content

# Save HTML file
html_file_path = 'great_expectations_report.html'
html_content = json_to_html(json_output)

with open(html_file_path, 'w') as file:
    file.write(html_content)

print(f"HTML report saved to {html_file_path}")

"""Using Data Profiler"""

!pip install sweetviz
import sweetviz as sv

# Generate the report
report = sv.analyze(df)
report.show_html('report.html')

!pip install dtale
import dtale

# Launch D-Tale server
dtale.show(df)

# Save the DataFrame to an HTML file
df.to_html('dataframe_view.html')

"""MetaData Management System Implementation"""

import json
import datetime

# Metadata for reports
report_metadata = {
    "great_expectations_report": {
        "type": "Great Expectations",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Great Expectations validation report for data quality.",
        "file_path": "great_expectations_report.html"
    },
    "sweetviz_report": {
        "type": "Sweetviz",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Sweetviz visualizations for exploratory data analysis.",
        "file_path": "report.html"
    },
    "dtale_report": {
        "type": "Dtale",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Dtale interactive data exploration report.",
        "file_path": "dataframe_view.html"
    },
    "pandas_profiling_report": {
        "type": "Pandas Profiling",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Pandas Profiling detailed exploratory data analysis.",
        "file_path": "pandas_profiling_report.html"
    }
}

# Save metadata to JSON
with open('report_metadata.json', 'w') as f:
    json.dump(report_metadata, f, indent=4)

print("Report metadata saved to 'report_metadata.json'")

"""Review Metadata"""

import json

# Load metadata from JSON file
with open('report_metadata.json', 'r') as f:
    metadata = json.load(f)

print("Loaded Metadata:")
print(json.dumps(metadata, indent=4))

[
    {
        "report_name": "sweetviz_report",
        "report_path": "path/to/sweetviz_report.html",
        "creation_date": "2024-07-24T12:00:00",
        "additional_info": {"details": "Exploratory Data Analysis"}
    }
]

import json

# Initialize a new metadata file
metadata_file = 'report_metadata.json'

# Create an empty metadata list
with open(metadata_file, 'w') as f:
    json.dump([], f, indent=4)

import datetime

def save_metadata(report_name, report_path, additional_info=None):
    metadata_file = 'report_metadata.json'

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        metadata_list = []

    # Create new metadata record
    new_metadata = {
        "report_name": report_name,
        "report_path": report_path,
        "creation_date": datetime.datetime.now().isoformat(),
        "additional_info": additional_info or {}
    }

    # Update or add new metadata
    for entry in metadata_list:
        if entry['report_name'] == report_name:
            entry.update(new_metadata)
            break
    else:
        metadata_list.append(new_metadata)

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
save_metadata('sweetviz_report', 'path/to/sweetviz_report.html', {"details": "Exploratory Data Analysis"})

def save_metadata(report_name, report_path, additional_info=None):
    metadata_file = 'report_metadata.json'

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        metadata_list = []

    # Print current metadata
    print("Current metadata:", metadata_list)

    # Create new metadata record
    new_metadata = {
        "report_name": report_name,
        "report_path": report_path,
        "creation_date": datetime.datetime.now().isoformat(),
        "additional_info": additional_info or {}
    }

    # Update or add new metadata
    for entry in metadata_list:
        if entry['report_name'] == report_name:
            entry.update(new_metadata)
            break
    else:
        metadata_list.append(new_metadata)

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
save_metadata('sweetviz_report', 'path/to/sweetviz_report.html', {"details": "Exploratory Data Analysis"})

def create_and_save_metadata(report_name, report_path, additional_info=None):
    # Create new metadata record
    new_metadata = {
        "report_name": report_name,
        "report_path": report_path,
        "creation_date": datetime.datetime.now().isoformat(),
        "additional_info": additional_info or {}
    }

    # Load existing metadata
    try:
        with open('report_metadata.json', 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        metadata_list = []

    # Append new record
    metadata_list.append(new_metadata)

    # Save updated metadata
    with open('report_metadata.json', 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
create_and_save_metadata('new_report.html', 'path/to/new_report.html', {"details": "New EDA report"})

def update_metadata_record(report_name, new_description=None):
    metadata_file = 'report_metadata.json'

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        metadata_list = []

    # Update existing record
    for entry in metadata_list:
        if entry['report_name'] == report_name:
            if new_description:
                entry['additional_info']['description'] = new_description
            break

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
update_metadata_record('new_report.html', 'Updated EDA report description')

def update_metadata_record(report_name, new_description=None):
    metadata_file = 'report_metadata.json'

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        metadata_list = []

    # Update existing record
    for entry in metadata_list:
        if entry['report_name'] == report_name:
            if new_description:
                entry['additional_info']['description'] = new_description
            break

    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
update_metadata_record('new_report.html', 'Updated EDA report description')

def retrieve_metadata(report_name):
    metadata_file = 'report_metadata.json'

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except FileNotFoundError:
        return None

    # Find metadata record
    for entry in metadata_list:
        if entry['report_name'] == report_name:
            return entry
    return None

# Example usage
metadata = retrieve_metadata('new_report.html')
print(metadata)

from google.colab import drive
drive.mount('/content/drive')

import os

# Define paths
metadata_dir = '/content/drive/My Drive/metadata/'
reports_dir = '/content/drive/My Drive/reports/'

# Create directories if they do not exist
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

import json

# Example metadata list
metadata_list = [
    {
        "report_name": "insurance_analysis.html",
        "path": "/content/drive/My Drive/reports/insurance_analysis.html",
        "description": "Exploratory Data Analysis on insurance data."
    }
]

# Define paths
metadata_path = os.path.join(metadata_dir, 'report_metadata.json')
report_path = os.path.join(reports_dir, 'insurance_analysis.html')

# Save metadata JSON file
with open(metadata_path, 'w') as f:
    json.dump(metadata_list, f, indent=4)

# Save your report (assuming `report` variable contains the HTML content)
with open(report_path, 'w') as f:
    f.write("<html><body>This is a sample report content.</body></html>")

# List files in the metadata directory
print(os.listdir(metadata_dir))

# List files in the reports directory
print(os.listdir(reports_dir))

import pandas_profiling
import sweetviz
import dtale

# Assuming you have generated these reports
# Replace these lines with actual report generation code
profile_report = pandas_profiling.ProfileReport(df)
sweetviz_report = sweetviz.analyze(df)
dtale_report = dtale.show(df)

# Save Pandas Profiling report
profile_report.to_file('/content/drive/My Drive/reports/insurance_analysis.html')

# Save Sweetviz report
sweetviz_report.show_html('/content/drive/My Drive/reports/sweetviz_report.html')

# Save Dtale report
# Assuming you want to save Dtale session; it doesn't have a direct save method
# You might need to export data manually if required.

# Check contents of the saved report files
with open('/content/drive/My Drive/reports/insurance_analysis.html', 'r') as f:
    print(f.read())

# Export the data to a CSV file
df.to_csv('/content/drive/My Drive/reports/insurance_data.csv', index=False)

# Example of incorporating data insights
with open('/content/drive/My Drive/reports/insurance_analysis.html', 'a') as file:
    file.write('<h2>Dtale Data Insights</h2>')
    file.write('<p>Data exported to CSV file: <a href="insurance_data.csv">Download CSV</a></p>')

# Update the metadata JSON file
def update_metadata_record(report_name, new_description):
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    for record in metadata_list:
        if record['report_name'] == report_name:
            record['description'] = new_description
            break

    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)

# Example usage
update_metadata_record('sweetviz_report', 'Updated Exploratory Data Analysis on insurance data.')

# Example script to retrieve metadata information
def get_metadata_info(report_name):
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    for record in metadata_list:
        if record['report_name'] == report_name:
            return record
    return None

# Example usage
report_info = get_metadata_info('sweetviz_report')
print(report_info)

# Example script to retrieve metadata information
def get_metadata_info(report_name):
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    for record in metadata_list:
        if record['report_name'] == report_name:
            return record
    return None

# Example usage
report_info = get_metadata_info('sweetviz_report')
print(report_info)

import json

# Verify the content of the metadata file
with open(metadata_path, 'r') as f:
    metadata_content = f.read()
    print(metadata_content)  # Check if the content is correct and valid JSON

[
    {
        "report_name": "sweetviz_report",
        "file_path": "path/to/sweetviz_report.html",
        "description": "Exploratory Data Analysis",
        "creation_date": "2024-07-24"
    }
]

def print_all_metadata():
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    for record in metadata_list:
        print(record)

print_all_metadata()

def get_metadata_info(report_name):
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    for record in metadata_list:
        if record.get('report_name') == report_name:
            return record
    return None

# Example usage
report_info = get_metadata_info('sweetviz_report')
print(report_info)

"""Data Cleaning"""

import pandas as pd

# Load the dataset
df = pd.read_csv('/content/insurance.csv')

# Example of handling missing values
df = df.fillna(method='ffill')  # Forward fill missing values

# Example of removing duplicates
df = df.drop_duplicates()

# Example of handling outliers (e.g., removing values beyond a certain range)
df = df[df['age'] < 100]  # Assuming age should be below 100

"""Satistical Analysis"""

# Descriptive Statistics
##  Summary Statistics:

import pandas as pd

# Load the dataset
df = pd.read_csv('/content/insurance.csv')

# Summary statistics
summary_stats = df.describe()
print(summary_stats)

## Frequency Counts

# Frequency counts for categorical columns
frequency_counts = df['sex'].value_counts()
print(frequency_counts)

#Distributions
## Visualize Distributions:

import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of numerical features
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot for identifying outliers
sns.boxplot(x=df['age'])
plt.title('Age Boxplot')
plt.show()

#Relationships
## Correlation Analysis:

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Heatmap of correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Pairwise Relationships:
# Pair plot to explore relationships
sns.pairplot(df[['age', 'bmi', 'charges']])
plt.title('Pairwise Relationships')
plt.show()

#scatterplot
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Age vs Charges')
plt.show()

#Hypothesis Testing
## Conduct Statistical Tests:

from scipy.stats import ttest_ind

# Example: t-test to compare age between two groups
group1 = df[df['sex'] == 'male']['age']
group2 = df[df['sex'] == 'female']['age']
t_stat, p_value = ttest_ind(group1, group2)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Example: Visualization of Charges by Age
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='charges', hue='sex')
plt.title('Charges vs Age')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(title='Sex')
plt.show()

"""Multiple Regression Analysis
Perform Multiple Regression Analysis
"""

import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/reports/insurance_data.csv')

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Prepare the data
X = df[['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]
X = sm.add_constant(X)  # Add a constant term for the intercept
y = df['charges']

# Perform the regression
model = sm.OLS(y, X).fit()
print(model.summary())

"""Analyzing Interaction Effects"""

# Add interaction term between age and smoker
df['age_smoker'] = df['age'] * df['smoker']

# Update the data
X = df[['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest', 'age_smoker']]
X = sm.add_constant(X)
y = df['charges']

# Perform the regression with interaction term
model = sm.OLS(y, X).fit()
print(model.summary())

"""Correlation analysis"""

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""Implement Predictive Models to Forecast Medical Costs
Data Preparation
Split Dataset into Training and Testing Sets:
"""

from sklearn.model_selection import train_test_split

# Split the data
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Normalizing the Data"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""Feature Selection"""

from sklearn.linear_model import LassoCV

# LassoCV for feature selection
lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
print("Selected features: ", X.columns[lasso.coef_ != 0])

"""Implementing Various ML Models & Evaluating Models"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42).fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate models
def evaluate_model(y_test, y_pred):
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R²: {r2_score(y_test, y_pred)}")

print("Linear Regression Performance:")
evaluate_model(y_test, y_pred_lr)

print("\nRandom Forest Performance:")
evaluate_model(y_test, y_pred_rf)

"""Fine Tune Hyperparameter"""

from sklearn.model_selection import GridSearchCV

# Example: Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

# Evaluate the best model
y_pred_best_rf = best_rf.predict(X_test_scaled)
print("Best Random Forest Performance:")
evaluate_model(y_test, y_pred_best_rf)

"""Interpreting the model using shap & Lime"""

!pip install shap
!pip install lime
import shap

# SHAP for Random Forest
explainer = shap.Explainer(best_rf)
shap_values = explainer.shap_values(X_test_scaled)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

pip install Flask openai

import pickle

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import shap
import numpy as np
import openai

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Add your OpenAI API key here
openai.api_key = 'your_openai_api_key'

@app.route('/')
def home():
    return render_template('/content/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    prediction = prediction[0]

    # Explain prediction using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(df_scaled)
    explanation_data = shap.Explanation(shap_values[0], base_values=explainer.expected_value, data=df_scaled[0]).data.tolist()

    # Generate explanation using GPT-3.5
    explanation_text = generate_explanation(explanation_data)

    return jsonify({
        'prediction': prediction,
        'explanation': explanation_text
    })

def generate_explanation(explanation_data):
    explanation_prompt = f"Given the SHAP values {explanation_data}, explain the prediction of medical insurance charges."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=explanation_prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    app.run(debug=True)