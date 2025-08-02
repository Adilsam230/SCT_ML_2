# SCT_ML_2
This Python script performs customer segmentation using the K-Means clustering algorithm. It groups customers into five distinct segments based on their annual income and spending score, then visualizes these groups in a scatter plot for easy analysis.

The program executes the following steps:
1. Reads customer information from a customer_data.csv file.
2. Prepares Data and selects the Annual_Income_k and Spending_Score features and standardizes them for the model (can be anything else depending on the contents present in the csv file).
3. Applies Clustering by using the K-Means algorithm to group the customers into 5 distinct clusters.
4. Visualizes Results by creating a scatter plot to visually represent the different customer segments, making it easy to see the groups.

You will need Python 3 and the following libraries:
1. Pandas
2. Matplotlib
3. Scikit-learn
