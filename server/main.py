from flask import Flask, request, render_template, jsonify
from ml.titanic import Titanic

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/fit")
def train():
    return render_template('train.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')
    # else:
    #     data = {
    #         'PassengerId': 891,
    #         'Pclass': 3,
    #         'Name': 'Kelly, Mr.James',
    #         'Sex':'male',
    #         'Age':34.5,
    #         'SibSp':0,
    #         'Parch':0,
    #         'Ticket':'330911',
    #         'Fare':7.82,
    #         'Cabin':'',
    #         'Embarked': 'Q',
    #         }
    #     data = Titanic.get_probability(data)
    #     response = {
    #         'prediction': data
    #     }
    #     return jsonify(response)

@app.route('/train_results')
def train_results():
    titanic = Titanic().fit()
    return render_template('train_results.html')


if __name__ == "__main__":
    app.run("127.0.0.1",debug=True)