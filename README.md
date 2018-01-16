# Predicting Electric Deficit In Grid

<b>What is this?</b>

This is my capstone project about making a simple prediction (0 or 1) about position of the deficit in the electric grid 
in the whole country using various data sources. Basically, the imbalance in the grid effects energy market prices and 
the target is to predict this imbalance. If there is enough electric production for the demand site, the number is "0" and 
not enough production is "1" and aim is to predict it.

<b>Why?</b>

Every powerplant gets a penalty for each imbalanced electricity(Kw) that generated. The brokers in the market tries to 
minimalize the penalty by making transactions in the market but they can learn the penalty five hours later from the imbalance 
happened. With the help of these project, brokers will have an idea about the position of the future hours.

<b>How?</b>

With he help of machine learning algorithms it was possible to make a simple prediction. The data being used in this project
are(Features):
- Hourly powerplant maintenance(as MW/h)
- Hourly powerplant failure(as MW/h)
- Hour as a number (0 - 23)
- Weekday as a number (1 - 7)
- Month as a number (1 - 12)
- Temperature (for each seven big city: SUM(population X city's temperature))

The data to predict(label):
- Position (0 or 1)

Scikit Learn is a gread open-source library about mmachine learning for a newbie like me. It is easy to use and easy to understand. Thats why i used it for my project.

I have used "Multilayer Perceptron Classifier" algorithm  to generate a model. I have seperated the data into two parts, 80% for trainig and 20%. i have set "100" hidden layer and  trained the model with the training data.Afther that i observed "Mean Squared Error(MSE)" during the backpropagation part.After a satisfying MSE, i stopped learning part and made a test with the test data. The score was approximately 74% and it was a satisfactory result for the people who needed it.

<b>Notes</b>

You can checkout the python code i added and project report for further information.
