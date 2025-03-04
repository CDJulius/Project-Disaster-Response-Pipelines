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
  

## Project Structure

Here’s how the project is organized:

### What’s in Each Folder?
- **`app/`**: Contains everything needed to run the web app.
  - **`templates/`**: HTML files for the app’s interface.
    - `go.html`: Shows the classification results.
    - `master.html`: The main page of the app.
  - **`run.py`**: Starts the Flask web app.

- **`data/`**: Contains the datasets and the script to clean and organize the data.
  - **`disaster_categories.csv`**: A dataset with categories for each message.
  - **`disaster_messages.csv`**: A dataset with disaster-related messages.
  - **`process_data.py`**: Cleans the data and saves it to a SQLite database.

- **`models/`**: Contains the script to train the machine learning model.
  - **`train_classifier.py`**: Trains the model and saves it as a `.pkl` file.

- **`README.md`**: This file! It explains everything about the project.

- **`requirements.txt`**: Lists all the Python libraries you need to install to run the project.

---

This structure keeps everything organized and makes it easy to find what you’re looking for. If you have any questions, feel free to ask!


