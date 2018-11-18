# Productionize your ML model

This is an ongoing project with an aim to have production level deployment code for Kaggle Titanic Dataset

If you are not aware of Kaggle Titanic dataset, I suggest you go and checkout the dataset from [here](https://www.kaggle.com/c/titanic)

The project is supposed to have the following features:-
- Open the browser and see the web page which has small introduction followed by two options (i) train and (ii) test
- Clicking on train will redirect to new page where the user can upload the Kaggle Training dataset.
- On hitting train the user will get training accuracy of the predictive model. On the backend there will be pickle file for trained mode and imputed values for prediction. In addition to that the data should be stored in ElasticSearch with probability of dying for each record in the training dataset.
- If the user clicks on predict function the page will redirect to a new page where the user can enter the details (features of Titanic dataset)
- User clicks on a button called `predict` which will give the probability of you surviving under those circumstance. In addition it should give other people (from training data) who the model gives same probability.
- The project should Dockerized
- The project should be hosted on AWS


This project uses Flask and Docker.


Common Issues:

- `OS Error: [Errro 99] Address already in use`: Just do `ps -fA | grep python` and kill the processes. 
View this [link](https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use) for detailed solution

## TODO: 
- Research on how good pandas are in Production Environment
- Write Production Level prediction Code (with Docstrings) without Pandas
- Write Production Level training code (with Docstrings) without Pandas

Some helpful links
- [Crete HTML form and insert data into database](https://www.c-sharpcorner.com/UploadFile/52bd60/create-an-html-form-and-insert-data-into-database162/)
- [Create a student registration form](https://www.roseindia.net/html/how-to-create-student-registration-form-with-html-code.shtml)
- [Create a web design](https://www.oreilly.com/library/view/learning-web-design/9781449337513/ch04.html)
