# Disaster Response Pipeline Project

## What's This About?
This project is part of the Udacity Data Scientist Nanodegree. It’s all about building a system to help emergency workers quickly sort and respond to disaster-related messages. Think of it like a smart assistant that reads messages (like tweets or texts) and figures out what kind of help is needed—food, water, medical aid, etc.

The project has three main parts:
1. **ETL Pipeline**: Cleans up messy data and organizes it into a database.
2. **ML Pipeline**: Trains a machine learning model to categorize messages.
3. **Web App**: A simple interface where you can type in a message and see how it gets categorized.


## Project Structure
The repository is organized as follows:

disaster-response-pipeline/
├── app/
│ ├── run.py # Flask app to run the web application
│ └── templates/ # HTML templates for the web app
│ ├── go.html # Result page
│ └── master.html # Main page
├── data/
│ ├── disaster_categories.csv # Dataset containing message categories
│ ├── disaster_messages.csv # Dataset containing disaster messages
│ └── process_data.py # Script to clean and save data to a database
├── models/
│ └── train_classifier.py # Script to train the ML model
├── README.md # This file
└── requirements.txt # List of dependencies


## How to Run the Project

### 1. Set Up the Environment
First, make sure you have Python 3 installed. Then, install the required dependencies

### Explanation of Dependencies
# pandas:

Used for data manipulation and loading the dataset from the SQLite database.

Version >=1.0.0 ensures compatibility with modern features.

# nltk:

Used for natural language processing tasks like tokenization, stopword removal, and lemmatization.

Version >=3.5 ensures access to the latest NLTK features and corpora.

# scikit-learn:

Used for machine learning tasks, including building the pipeline, training the model, and evaluating performance.

Version >=0.24.0 ensures compatibility with the latest scikit-learn features.

# sqlalchemy:

Used to connect to and query the SQLite database.

Version >=1.4.0 ensures compatibility with modern database features.

# joblib:

Used to save and load the trained model.

Version >=1.0.0 ensures compatibility with the latest joblib features.

### 2. Run the ETL Pipeline
To clean the data and save it to a SQLite database, run the following command:

# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

This script will:

Load the disaster_messages.csv and disaster_categories.csv files.

Clean and merge the datasets.

Save the cleaned data to DisasterResponse.db.

### 3. Train the Machine Learning Model
Next, train the machine learning model by running:

# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

This script will:

Load the cleaned data from the SQLite database.

Train a machine learning model to classify messages into categories.

Save the trained model as classifier.pkl.

### 4. Run the Web App
Finally, to launch the web app, run:


python app/run.py
This will start a Flask web server. Open your browser and go to http://0.0.0.0:3001/ to view the app. You can input a message, and the app will classify it into one or more of the 36 categories.

Files in the Repository
data/process_data.py
This script handles the ETL pipeline:

load_data: Loads the messages and categories datasets and merges them.

clean_data: Cleans the data by splitting categories, converting values to binary, and removing duplicates.

save_data: Saves the cleaned data to an SQLite database.

models/train_classifier.py
This script trains the machine learning model:

Loads the cleaned data from the database.

Preprocesses the text data using NLP techniques.

Trains a classifier using a pipeline with TF-IDF and a multi-output classifier.

Saves the trained model to a .pkl file.

app/run.py
This script runs the Flask web app:

Loads the trained model and database.

Provides a user interface where users can input messages and see classification results.

app/templates/
This folder contains the HTML templates for the web app:

master.html: The main page where users can input messages.

go.html: The result page that displays the classification results.

Dependencies
The project requires the following Python libraries:

pandas

numpy

scikit-learn

nltk

sqlalchemy

flask

plotly

You can install all dependencies by running:

bash
Copy
pip install -r requirements.txt
Summary
This project provides a pipeline for processing disaster-related messages and classifying them into categories. The ETL pipeline cleans and stores the data, while the machine learning model classifies the messages. The web app allows users to interact with the model and see classification results in real-time.

Feel free to explore the code, run the scripts, and play around with the web app!

