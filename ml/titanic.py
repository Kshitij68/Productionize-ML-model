from util import get_logger
import pickle
import os
import csv
import statistics
from sklearn.ensemble import RandomForestClassifier

# TODO: Add Unit Tests for all functions
# TODO: Add PCA
logger = get_logger('Machine Learning')


class Titanic():

    def __init__(self):
        self.pclass_mode = None
        self.name = None
        self.sex_mode = None
        self.age_mean = None
        self.sibsp_mode = None
        self.parch__mode = None
        self.fare_mean = None
        self.embarked_mode = None

    @staticmethod
    def load_training_data(path):
        """
        Load the training data and save it in a list of dictionaries
        :param path:
        :return:
        """
        basedir = os.path.dirname(__file__)
        try:
            path = os.path.join(basedir,'data/train.csv')
        except:
            path = os.path.join(basedir,path)
        csv_file = open(path,'r',newline='',encoding='latin-1')
        csv_reader = csv.reader(csv_file, delimiter=',')
        train_x = list()
        train_y = list()
        for index,value in enumerate(csv_reader):
            if index == 0:
                continue
            else:
                train_x.append({
                    'PassengerId': value[0],
                    'Pclass': value[2],
                    'Name': value[3], # Perform One-Hot Encoding
                    'Sex': value[4], # Perform One-Hot Encoding
                    'Age': value[5], # Get Age Bins
                    'SibSp': value[6],
                    'Parch': value[7],
                    'Ticket': value[8],
                    'Fare': value[9],
                    'Cabin': value[10],
                    'Embarked': value[11]
                })
                train_y.append(value[1])
        return train_x, train_y

    @staticmethod
    def load(filename):
        basedir = os.path.dirname(__file__)
        model_path = os.path.join(basedir, filename)
        pickle_in = open(model_path, "rb")
        return pickle.load(pickle_in)

    @staticmethod
    def save(filename,file):
        basedir = os.path.dirname(__file__)
        filename = os.path.join(basedir, filename)
        data = open(filename, 'wb')
        pickle.dump(file, data)
        data.close()

    @staticmethod
    def get_stats(data, key, typ, convert_to_float = False):
        if convert_to_float:
            values = list()
            for value in data:
                try:
                    values.append(float(value[key]))
                except:
                    logger.info("Unable to convert {} to float, skipping it".format(value[key]))
        else:
            values = [value[key] for value in data]
        if typ == 'median':
            return statistics.median(values)
        elif typ == 'mean':
            return statistics.mean(values)
        elif typ == 'mode':
            return statistics.mode(values)

    @staticmethod
    def drop_values(data, keys):
        """
        Drop the keys which are not useful in prediction
        """
        for dictionary in data:
            for key in keys:
                if key not in dictionary.keys():
                    raise Exception('The dictionary {} does not have key {}'.format(dictionary, key))
                del dictionary[key]
        return data

    @staticmethod
    def encode_embarked(embarked):
        if embarked == 'S':
            return [1, 0, 0]
        elif embarked == 'C':
            return [0, 1, 0]
        elif embarked == 'Q':
            return [0, 0, 1]

    @staticmethod
    def encode_gender(gender):
        if gender == 'male':
            return [1, 0]
        return [0, 1]

    @staticmethod
    def encode_title(title):
        if title == 'Mr.':
            return [1, 0, 0, 0, 0]
        elif title == 'Mrs':
            return [0, 1, 0, 0, 0]
        elif title == 'Miss':
            return [0, 0, 1, 0, 0]
        elif title == 'Master':
            return [0, 0, 0, 1, 0]
        return [0, 0, 0, 0, 1]

    @staticmethod
    def fill(string, imputed_value):
        if len(str(string)):
            return string
        return imputed_value

    def fill_na(self,data):
        for dictionary in data:
            dictionary['Pclass'] = self.fill(dictionary['Pclass'], self.pclass_mode)
            dictionary['Name'] = self.fill(dictionary['Name'], self.name)
            dictionary['Sex'] = self.fill(dictionary['Sex'], self.sex_mode)
            dictionary['Age'] = self.fill(dictionary['Age'], self.age_mean)
            dictionary['SibSp'] = self.fill(dictionary['SibSp'], self.sibsp_mode)
            dictionary['Parch'] = self.fill(dictionary['Parch'], self.parch__mode)
            dictionary['Fare'] = self.fill(dictionary['Fare'], self.fare_mean)
            dictionary['Embarked'] = self.fill(dictionary['Embarked'], self.embarked_mode)
        return data

    @staticmethod
    def get_title(string):
        string = str(string)
        if not len(string):
            return 'other'
        string = string.split(", ")
        if len(string) < 2:
            return 'other'
        string = string[1]
        string = string.split(".")[0]
        if string in ['Mr', 'Miss', 'Mrs', 'Master']:
            return string
        elif string == ['Lady', 'Ms', 'Mlle']:
            return 'Miss'
        return 'other'

    def create_features(self,data):
        for dictionary in data:
            dictionary['Name'] = self.get_title(dictionary['Name'])
            dictionary['FamilySize'] = dictionary['SibSp'] + dictionary['Parch'] + 1
            if dictionary['FamilySize'] == 1:
                dictionary['IsAlone'] = 1
            else:
                dictionary['IsAlone'] = 0

    def encode(self,data):
        for dictionary in data:
            embarked_array = self.encode_embarked(dictionary['Embarked'])
            dictionary['Embarked_S'] = embarked_array[0]
            dictionary['Embarked_C'] = embarked_array[1]
            dictionary['Embarked_Q'] = embarked_array[2]

            gender_array = self.encode_gender(dictionary['Sex'])
            dictionary['sex_M'] = gender_array[0]
            dictionary['sex_F'] = gender_array[1]

            title_array = self.encode_title(dictionary['Name'])
            dictionary['Name_Mr'] = title_array[0]
            dictionary['Name_Mrs'] = title_array[1]
            dictionary['Name_Miss'] = title_array[2]
            dictionary['Name_Master'] = title_array[3]
            dictionary['Name_Misc'] = title_array[4]
        return data

    @staticmethod
    def convert_to_float(data):
        for dictionary in data:
            keys = dictionary.keys()
            for key in keys:
                dictionary[key] = float(dictionary[key])
        return data

    @staticmethod
    def dict_to_array(data):
        if len(data) and type(data) == list and type(data[0] == dict):
            keys = sorted(list(data[0].keys()))
        else:
            return []
        array = list()
        for dictionary in data:
            arr = list()
            for key in keys:
                arr.append(dictionary[key])
            array.append(arr)
        return array

    def save_imputed_values(self,data):
        path = 'models/'
        self.pclass_mode = self.get_stats(data, 'Pclass', 'mode')
        self.save(path + 'pclass_mode.sav', self.pclass_mode)

        self.name = 'Mathur, Mr. Kshitij'
        self.save(path + 'name.sav', self.name)

        self.sex_mode = self.get_stats(data, 'Sex', 'mode')
        self.save(path + 'sex_mode.sav', self.sex_mode)
        self.age_mean = self.get_stats(data, 'Age', 'mean', convert_to_float=True)
        self.save(path + 'age_mean.sav', self.age_mean)

        self.sibsp_mode = self.get_stats(data, 'SibSp', 'mode', convert_to_float=True)
        self.save(path + 'sibsp_mode.sav', self.sibsp_mode)

        self.parch__mode = self.get_stats(data, 'Parch', 'mode', convert_to_float=True)
        self.save(path + 'parch_mode.sav', self.parch__mode)

        self.fare_mean = self.get_stats(data, 'Fare', 'mean', convert_to_float=True)
        self.save(path + 'fare_mean.sav', self.fare_mean)

        self.embarked_mode = self.get_stats(data, 'Embarked', 'mode')
        self.save(path + 'embarked_mode.sav', self.embarked_mode)

    def load_imputed_values(self):
        path = 'models/'
        self.pclass_mode = self.load(path + 'pclass_mode.sav')
        self.name = self.load(path + 'name.sav')
        self.sex_mode = self.load(path + 'sex_mode.sav')
        self.age_mean = self.load(path + 'age_mean.sav')
        self.sibsp_mode = self.load(path + 'sibsp_mode.sav')
        self.parch__mode = self.load(path + 'parch_mode.sav')
        self.fare_mean = self.load(path + 'fare_mean.sav')
        self.embarked_mode = self.load(path + 'embarked_mode.sav')

    def fit(self, path = ''):

        train_x, train_y = self.load_training_data(path)

        self.save_imputed_values(train_x)

        train_x = self.drop_values(train_x, ['PassengerId', 'Cabin', 'Ticket'])
        train_x = self.fill_na(train_x)
        train_x = self.encode(train_x)
        train_x = self.drop_values(train_x, ['Embarked', 'Sex', 'Name'])
        train_x = self.convert_to_float(train_x)
        train_x = self.dict_to_array(train_x)

        rf = RandomForestClassifier()
        rf.fit(train_x, train_y)
        self.save('models/model.sav', rf)

    def get_probability(self, test):
        """
        Function that will give the probability of someone dying under the entered circumstances
        :param test: dictionary containing the following keys:
            PassengerId: int
            Pclass: int 
            Name: str 
            Sex: str
            Age: int
            SibSp: int
            Parch: int
            Ticket: str
            Fare: float
            Cabin: str
            Embarked: str
        :return: probability: float
            Probability of death
        """

        self.load_imputed_values()

        test = self.drop_values(test, ['PassengerId', 'Cabin', 'Ticket'])
        test = self.fill_na(test)
        test = self.encode(test)
        test = self.drop_values(test, ['Embarked', 'Sex', 'Name'])
        test = self.convert_to_float(test)
        test = self.dict_to_array(test)
        model = self.load('models/model.sav')

        probability = model.predict_proba(test)

        return probability


# if __name__ == "__main__":
#     titanic = Titanic()
# #    titanic.fit('data/')
#     print(titanic.get_probability([{'PassengerId': 1, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male',
#                                    'Age': 22, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': '',
#                                     'Embarked': 'S'}]))
