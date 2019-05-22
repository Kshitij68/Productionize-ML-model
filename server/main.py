from flask import Flask, request, render_template, jsonify
from ml.titanic import Titanic

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/fit")
def train():
    return render_template('train.html')


@app.route('/predict',methods = ['GET'])
def predict():
    return render_template('predict.html')

@app.route('/train_results')
def train_results():
    titanic = Titanic()
    titanic.fit()
    return render_template('train_results.html')


if __name__ == "__main__":
    app.run("127.0.0.1",debug=True)