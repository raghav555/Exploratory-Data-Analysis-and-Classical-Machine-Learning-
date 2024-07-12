# Exploratory-Data-Analysis-and-Classical-Machine-Learning
This is a project where in i did exploratory Data Analysis on publicly available dataset (data cleaning, data preparation, filling Nan Values) and applied Classical Machine Learning Techniques such as Linear Regression, Logistic Regression, K neighbors regressor and deep learning techniques and tuned various hyperparameters.

Part 1: Data Analysis 
The dataset that we are using in this project is winequality-red.csv. Since the dataset is semicolon separated instead of comma separated we use a different separator i.e. semicolon(;) instead of the default(,) 

There are a total of 12 columns and 1599 rows in the dataset

The dataset has integer value for quality column whereas float64 for all other columns 
#NULL Stats 
There are no null values in the dataset as given by the below command
Some common statistics of the dataset are: 
Mean

Standard Deviation

The unique values of every column are:

3 Visualizations of the dataset are: 
![Screenshot 2024-07-12 at 10 05 32 AM](https://github.com/user-attachments/assets/0ea5913b-1fbd-4fec-b87a-c8bd44922273)

The above scatter plot shows alcohol level wrt density. From the scatter plot it can be observed that the lesser the alcohol level the more dense the wine is which is true since density of alcohol is less than that of water. Hence the lesser the alcohol is in the wine the more the density of that wine.

![Screenshot 2024-07-12 at 10 06 29 AM](https://github.com/user-attachments/assets/0c0bb2a3-da80-46f3-85bf-2c2149af805e)
The above plot shows variation of pH level wrt alcohol. From the graph it can be observed that All wines are acidic (pH < 7) and most wines have pH around 3.2-3.6 mark

![Screenshot 2024-07-12 at 10 06 44 AM](https://github.com/user-attachments/assets/a145aa88-c1a5-49c7-9bb3-cdd88bd138df)

The above graph shows the variation of alcohol wrt to quality. From the graph it can be observed that generally the lesser the alcohol the lesser is the quality and vice versa.


**Part 2: ML Analysis **
Analysis 1: Linear Regression 
When modeling the relationship between a scalar answer and one or more explanatory factors, linear regression is a linear method (also known as dependent and independent variables). 
We took density as feature and quality as the target variable for linear regression and dropped any null rows. 
We then reshaped our features and target variable and split it into training and testing dataset (80/20) and applied linear regression to get accuracy of 0.03 which is 3%. Getting such a low accuracy implies that the variables are highly uncorrelated.
![Screenshot 2024-07-12 at 10 07 32 AM](https://github.com/user-attachments/assets/3a6f502b-27dd-47c8-804b-670a8d676e2b)


Analysis 2: Logistic Regression 
Using one or more independent variables and a dependent variable together, logistic regression is used to assess the relationship between them. 
We used logistic regression on the wine dataset with the features as all columns except quality and target variable as quality. We again split the data into training and testing dataset (80/20) We then trained with max iteration = 1500

Then we used logistic regression to predict accuracy on the testing data and got an accuracy of 0.57 

We plotted the below plot to show how good our model is predicting on the actual data by logistic regression 
![Screenshot 2024-07-12 at 10 07 52 AM](https://github.com/user-attachments/assets/067aa113-a176-43ce-ade6-134bd0c275d6)


Analysis 3: K Neighbors Regressor 
The k-nearest neighbors technique employs proximity to classify or anticipate how a single data point will be grouped. It is a non-parametric, supervised learning classifier. 
We took the first 10 columns as our features and quality as the target variable. We then split the dataset into 80/20 training and testing data.
We then standardized our training and testing dataset. 
We then applied the KNN Model on our dataset and got the following results for k = 1 to 50
Since the last value for k = 49 is 0.455 thus the accuracy of our model is 0.455 by KNN 
Comparing the result for the 3 Machine Learning Models:
Comparison of the 3 machine learning models: 
Using the above 3 models i.e. KNN, Linear Regression and Logistic Regression we got the accuracy of 0.45, 0.43 and 0.57 respectively. Out of the three models Logistic Regression has the highest accuracy. Logistic regression typically outperforms linear regression and KNN since all features are independent of one another. When there is some correlation between the characteristics, linear regression and KNN perform at their best. However, correlation between input features is not a requirement for logistic regression. This explains why, out of all the models we used, Logistic regression had the highest accuracy.


**Part 3: Deep learning **
We trained our model on a neural network with the layers 64 and activation function as softmax. We used adam optimizer and loss as sparse_categorical_crossentropy. 
We then trained our model on 5 epochs and got the following loss and accuracy. We then predicted on our testing data to get the following loss and accuracy 
Hyperparameters Tuning: 
1.Changing the number of epochs 
Epochs = 10

Epochs = 100 
We see that increasing the number of epochs in our neural network leads to lower loss and better accuracy 
2. Changing the optimizer 
Optimizer = adam
Accuracy = 0.57 

Optimizer = sgd 
Accuracy = 0.50 

Optimizer = Adadelta 
Accuracy = 0.43 
Optimizer = Adagrad 
Accuracy = 0.51 

Optimizer = Adamax 
Accuracy = 0.56
Out of the optimizers used adam gives the highest accuracy 
3. Changing the number of dimensions of a layer in neural network 
Layers = 64 
Accuracy = .59 

![Screenshot 2024-07-12 at 10 08 24 AM](https://github.com/user-attachments/assets/47867250-aceb-4116-bfb5-b30989fd92b8)


Layers = 32 
Accuracy = 0.56 

Layers = 16 
Accuracy = 0.52

Thus we see that increasing the number of dimensions of a layer in the neural network leads to better accuracy 
References: 
Pandas documentation : https://pandas.pydata.org/docs/index.html Sklearn documentation : https://scikit-learn.org/stable/ Python documentation : https://docs.python.org/3/ 
Matplotlib documentation : https://matplotlib.org/stable/index.html https://en.wikipedia.org/wiki/Linear_regression


