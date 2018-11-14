# Productionize your ML model

This is an ongoing project with an aim to have production level deployment code for Kaggle Titanic Dataset

If you are not aware of Kaggle Titanic dataset, I suggest you go and checkout the dataset from [here](https://www.kaggle.com/c/titanic)

This project uses Flask and Docker.

Common Issues:

- `OS Error: [Errro 99] Address already in use`: Just do `ps -fA | grep python` and kill the processes. 
View this [link](https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use) for detailed solution

## TODO: 
- Research on how good pandas are in Production Environment