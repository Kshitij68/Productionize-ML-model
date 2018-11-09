from flask import Flask
from flask import jsonify
from predict import get_probability

app = Flask(__name__)

@app.route("/")
def hello():
    return 'Hello, Do you think you would have survived the Titanic Disaster?'

@app.route('/predict')
def predict():
    data = {
    'PassengerId':891,
    'Pclass':3,
    'Name':'Kelly, Mr.James',
    'Sex':'male',
    'Age':34.5,
    'SibSp':0,
    'Parch':0,
    'Ticket':'330911',
    'Fare':7.82,
    'Cabin':'',
    'Embarked': 'Q',
    }
    data = get_probability(data).tolist()
    response = {
        'prediction':data
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)