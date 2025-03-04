# Disaster Response Pipeline Project

## What's This About?
This project is part of the Udacity Data Scientist Nanodegree. It’s all about building a system to help emergency workers quickly sort and respond to disaster-related messages. Think of it like a smart assistant that reads messages (like tweets or texts) and figures out what kind of help is needed—food, water, medical aid, etc.

The project has three main parts:
1. **ETL Pipeline**: Cleans up messy data and organizes it into a database.
2. **ML Pipeline**: Trains a machine learning model to categorize messages.
3. **Web App**: A simple interface where you can type in a message and see how it gets categorized.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [How the Project is Organized](#how-the-project-is-organized)
3. [How to Run the Project](#how-to-run-the-project)
4. [Example Usage](#example-usage)
5. [Try It Out!](#try-it-out)
6. [License](#license)

---

## Getting Started
### What You’ll Need
- **Python 3.x**: Make sure you have Python installed.
- **Libraries**: Install the required libraries
-   - **Pandas**: For data manipulation and analysis.
   - **NumPy**: For numerical computations.
   - **Scikit-learn**: For machine learning.
   - **SQLAlchemy**: For working with databases.
   - **Flask**: For running the web app.
   - **NLTK**: For natural language processing.
   - **Joblib**: For saving and loading the machine learning model.
  

disaster-response-pipeline/
├── app/
│   ├── templates/          # HTML files for the web app
│   │   ├── go.html         # Page that shows classification results
│   │   └── master.html     # Main page of the app
│   └── run.py              # Script to start the web app
├── data/
│   ├── disaster_categories.csv  # Dataset with message categories
│   ├── disaster_messages.csv    # Dataset with disaster messages
│   └── process_data.py          # Script to clean and save data
├── models/
│   └── train_classifier.py      # Script to train the ML model
├── README.md                    # This file!
└── requirements.txt             # List of Python libraries needed
