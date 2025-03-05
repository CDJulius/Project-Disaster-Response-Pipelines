# models/train_classifier.py
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
        database_filepath (str): Path to the SQLite database.
    
    Returns:
        X (pd.Series): Messages (features).
        Y (pd.DataFrame): Categories (targets).
        category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize text.
    
    Args:
        text (str): Input text.
    
    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline.
    
    Returns:
        model (GridSearchCV): Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance.
    
    Args:
        model: Trained model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): Test targets.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.
    
    Args:
        model: Trained model.
        model_filepath (str): Path to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...')
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()