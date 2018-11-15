import logging
import pandas as pd
import pickle
import os
import csv
from ml.feature_engineering import drop_values,get_age_bins,get_fare_bins
from sklearn.ensemble import RandomForestClassifier

# ADD TRAINING SET
# TODO: ADD Loggers
# TODO: Add Unit Tests for all functions
# TODO: Add PCA


def load(filename):
    pickle_in = open(filename, "rb")
    return pickle.load(pickle_in)

def encode_embarked(embarked):
    if embarked == 'S':
        return [1,0,0]
    elif embarked == 'C':
        return[0,1,0]
    elif embarked == 'Q':
        return [0,0,1]


def encode_gender(gender):
    if gender == 'male':
        return [1,0]
    return [0,1]


def encode_title(title):
    if title == 'Mr.':
        return [1,0,0,0,0]
    elif title == 'Mrs':
        return [0,1,0,0,0]
    elif title == 'Miss':
        return [0,0,1,0,0]
    elif title == 'Master':
        return [0,0,0,1,0]
    return [0,0,0,0,1]


def encode_alone(alone):
    if alone == 0:
        return [1,0]
    return [0,1]


def get_features(features):
    """
    To get the features from the values
    Parameters
    -----------
    features    : dictionary containing the following keys
        PassengerId:
        Pclass:
        Name:
        Sex:
        Age:
        SibSp:
        Parch:
        Ticket:
        Fare:
        Cabin:
        Embarked:
    """
    features = pd.DataFrame([features])
    median_age = load('median_age.sav')
    embarked_mode = load('embarked_mode.sav')
    fare_median = load('median_fare.sav')
    title_names = load('title_names.sav')

    features['Age'].fillna(median_age, inplace=True)

    features.drop(['Cabin'],axis=1,inplace=True)

    features.fillna(embarked_mode, inplace = True)

    features.fillna(fare_median, inplace = True)

    features.drop(['Ticket','PassengerId'],axis=1,inplace=True)

    features['Title'] = features['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    features.drop(['Name'],axis=1,inplace=True)
    features[features['Title'] == 'Ms']['Title'] = 'Miss'
    features[features['Title'] == 'Mlle']['Title'] = 'Miss'
    features['Title'] = features['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    features['FamilySize'] = features['SibSp'] + features['Parch'] + 1
    features['IsAlone'] = 1
    features['IsAlone'].loc[features['FamilySize'] > 1] = 0

    features['Age'] = features['Age'].apply(lambda x: get_age_bins(x))

    features['Fare'] = features['Fare'].apply(lambda x: get_fare_bins(x))

    features.Embarked = features.Embarked.apply(lambda x: encode_embarked(x))
    features['Embarked_S'] = features.Embarked.apply(lambda x: x[0])
    features['Embarked_C'] = features.Embarked.apply(lambda x: x[1])
    features['Embarked_Q'] = features.Embarked.apply(lambda x: x[2])
    features.drop(['Embarked'], axis=1, inplace=True)

    features['Sex'] = features.Sex.apply(lambda x: encode_gender(x))
    features['Sex_male'] = features.Sex.apply(lambda x: x[0])
    features['Sex_female'] = features.Sex.apply(lambda x: x[1])
    features.drop(['Sex'], axis=1, inplace=True)

    features['Title'] = features.Title.apply(lambda x: encode_title(x))
    features['Title_Mr'] = features.Title.apply(lambda x: x[0])
    features['Title_Mrs'] = features.Title.apply(lambda x: x[1])
    features['Title_Miss'] = features.Title.apply(lambda x: x[2])
    features['Title_Master'] = features.Title.apply(lambda x: x[3])
    features['Title_Misc'] = features.Title.apply(lambda x: x[4])
    features.drop(['Title'], axis=1, inplace=True)

    features.IsAlone = features.IsAlone.apply(lambda x: encode_alone(x))
    features['IsAlone_0'] = features.IsAlone.apply(lambda x: x[0])
    features['IsAlone_1'] = features.IsAlone.apply(lambda x: x[1])
    features.drop(['IsAlone'], axis=1, inplace=True)
    return features


def load_training_data(path):
    """
    Load the training data and save it in a list of dictionaries
    :param path:
    :return:
    """
    basedir = os.path.dirname(__file__)
    path = os.path.join(basedir,'data/train.csv')
    csv_file = open(path,'r',newline='',encoding='latin-1')
    csv_reader = csv.reader(csv_file, delimiter=',')
    train_x = list()
    train_y = list()
    for index,value in enumerate(csv_reader):
        if index == 0:
            continue
        else:
            train_x.append({
                'PassengerId': value[0], # Dropped the column
                'Pclass': value[2], # Null values filled with mode #
                'Name': value[3], # Raise an error if Null values are entered,
                'Sex': value[4],
                'Age': value[5], # Null values filled with mode #
                'SibSp': value[6],
                'Parch': value[7],
                'Ticket': value[8], # Dropped the column
                'Fare': value[9], # Null values filled with median
                'Cabin': value[10], # Dropped the column
                'Embarked': value[11] # Null values filled with mode
            })
            train_y.append({
                'Survived':value[1]
            })
    return train_x, train_y


# def train(path):
#     train_x, train_y = load_training_data(path)
#     train = drop_values(train_x,['PassengerId', 'Cabin', 'Ticket'])
#     train = get_median()
