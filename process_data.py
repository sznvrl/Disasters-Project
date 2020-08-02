import sys
import pandas as pd
from sqlalchemy import create_engine

categories_filepath = "/home/workspace/models/data/disaster_categories.csv"
messages_filepath = "/home/workspace/models/data/disaster_messages.csv"

def load_data(messages_filepath, categories_filepath):
    
    """
    load dataset
    
    Parameters:
    messages_filepath: filepath for messages data
    categories_filepath: filepath for categories data
    
    Return:
    df: merge categories and messages data sets
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="inner",on='id')
    
    return df
    


def clean_data(df):
    
    """
    clean dataset
    
    Parameters
    df= merge messages+categories data set
    
    Returns
    df_new= cleaned data set
    
    
    
    """ 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    
    categories =  categories["categories"].str.split(';', expand=True)
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda r: r[:-2]).iloc[0, :].tolist()
    
    #get categories names
    categories.columns = category_colnames
    
    for column in categories:
        # create each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        #convert column  to numeric
        categories[column] = categories[column].astype(int)
        
    df=df.drop("categories", axis=1)
    frames=[df, categories]
    df_new = pd.concat(frames, axis=1)
    
    #drop duplicates according to message and category_colnames
    df_new= df_new.drop_duplicates(subset='message')
    df_new=df_new.dropna(subset=category_colnames)
    
    return df_new


def save_data(df, database_filename):
    engine = create_engine("sqlite:///"+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    
    return engine


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        print(database_filepath,messages_filepath, categories_filepath )
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()