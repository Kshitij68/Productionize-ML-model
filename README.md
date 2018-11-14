# Productionize your ML model

This is an ongoing project with an aim to have production level deployment code for Kaggle Titanic Dataset

If you are not aware of Kaggle Titanic dataset, I suggest you go and checkout the dataset from [here](https://www.kaggle.com/c/titanic)

This project uses Flask and Docker.

Common Issues:

- `OS Error: [Errro 99] Address already in use`: Just do `ps -fA | grep python` and kill the processes. 
View this [link](https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use) for detailed solution

## TODO: 
- Research on how good pandas are in Production Environment
- Get all the files from the Jupyter Notebook
- Add Jupyter Notebook in the gitignore
- Write Production Level prediction Code (with Docstrings) without Pandas
- Write Production Level training code (with Docstrings) without Pandas

Some helpful links
- [Crete HTML form and insert data into database](https://www.c-sharpcorner.com/UploadFile/52bd60/create-an-html-form-and-insert-data-into-database162/)
- [Create a student registration form](https://www.roseindia.net/html/how-to-create-student-registration-form-with-html-code.shtml)
- [Create a web design](https://www.oreilly.com/library/view/learning-web-design/9781449337513/ch04.html)
