import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import string
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.model_selection  import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV


def load_data(db_path):
    """
    Function: load data from database and return X and y.
    Args:
      db_path(str): database file name included path
      tablename:(str): table name in the database file.
    Return:
      X(pd.DataFrame): messages for X
      Y(pd.DataFrame): labels part in messages for y
    """
    
    # load data from database
    engine = create_engine("sqlite:///"+db_path)
    #read "messages" table
    df = pd.read_sql("SELECT * FROM messages", engine)
                           
    # split dataset to two subsets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    #get columns names
    columns_names = Y.columns.tolist()
    
    return  X, Y, columns_names



#get necessery inputs for cleaning messages
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
remove_punc_table = str.maketrans('', '', string.punctuation)

def tokenize(text):
    
    """
    tokenize and cleaning text
    
    Parameters:
    text: Give text which is tokenized
    
    Returns:
    word: text split to words
   
    """

    #remove punctuation
    text = text.translate(remove_punc_table).lower()
    
    #tokenize text
    tokens = nltk.word_tokenize(text)
    
    #lemmatize and remove stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model( ):
    """
    create pipeline and parameters for model
    find best model's parameters 
    
    Arg:
    NaN
    
    Return:
    grid_search: optimal model which have best model's parameters
   
    """
    
    pipeline = Pipeline([
                    
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    parameters = {'clf__estimator__max_leaf_nodes': [3, 5],
                   'clf__estimator__min_samples_leaf': [8, 10]}
    
   

    grid_search = GridSearchCV(pipeline, parameters,verbose= 10, n_jobs =-1)
   
    return grid_search




def evaluate_model(model, X_test, Y_test, columns_names):
    
     """
     Calculate model's score; accuracy, precision, recall and f1 score
    
     Parameters
     model: determine model and model's parameters
     Y_test: test set/categories
     X_test: test_set/messages-text
     column_names: categories names
    
     Returns
     df_score: dataframe/evaluation metrics
     
     """
     score = []
     y_pred = np.array(model.predict(X_test))
    
    
     #find accuracy, precision, recall and f score for each categories
     for i in range(len(columns_names)):
         accuracy = accuracy_score(Y_test.iloc[:, i], y_pred[:, i])
         precision = precision_score(Y_test.iloc[:, i], y_pred[:, i], average=None)
         recall = recall_score(Y_test.iloc[:, i], y_pred[:, i], average=None)
         f1 = f1_score(Y_test.iloc[:, i], y_pred[:, i], average=None)
        
         score.append([accuracy, precision, recall, f1])
    
     #Create dataframe 
     df_score = pd.DataFrame(score, index = columns_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    
     return print(df_score)
   



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
       
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()