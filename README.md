# Spaceship-Titanic-Kaggle-
Titanic Passenger Analysis & Transport Prediction
This project performs analysis and machine learning modeling on a fictional space voyage dataset (inspired by the Kaggle Spaceship Titanic competition). It includes data cleaning, exploratory data analysis (EDA), visualizations, and classification models to predict whether a passenger was transported.

Dataset Used
Filename: test.csv
Attributes include:
Demographic features: Age, HomePlanet, Name
Spending details: FoodCourt, Spa, ShoppingMall, RoomService, VRDeck
Travel info: Destination, CryoSleep, Cabin, etc.

Note: If the dataset does not contain the Transported column, it will be randomly generated for demonstration purposes.

Features
Data Preprocessing
Dropping irrelevant columns: PassengerId, Name, Cabin
Handling missing values:
Numeric columns: Mean imputation
Categorical columns: Most frequent value imputation
Encoding categorical variables using LabelEncoder

Exploratory Data Analysis (EDA)
Distribution plots of numerical features
Count plots for categorical features
Boxplots and violin plots
Correlation heatmap
Spending pattern analysis by destination
Pie chart of home planet distribution

Interactive Visualizations (Plotly)
Scatter plot: Age vs. VRDeck, colored by destination
Sunburst chart: HomePlanet → CryoSleep → Destination

Machine Learning Models
Random Forest Classifier
Logistic Regression
Performance evaluated using:
Accuracy
Precision, Recall, F1-score (via classification_report)
Confusion matrix

Libraries Used

Edit
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn

How to Run
Clone the repository:
bash
git clone https://github.com/yourusername/spaceship-titanic-analysis.git
cd spaceship-titanic-analysis
Install the required packages:

bash

pip install -r requirements.txt
Run the script (in a Jupyter Notebook, Google Colab, or Python environment):

bash

python analysis.py
Project Structure
bash

├── analysis.py          # Main Python script
├── test.csv             # Input dataset
├── README.md            # Project description
├── requirements.txt     # List of required libraries

Future Enhancements
Implement hyperparameter tuning using GridSearchCV or RandomizedSearchCV
Test additional classifiers such as XGBoost or SVM
Deploy using Streamlit or Flask for a user-friendly interface

Author
Priya R.
www.linkedin.com/in/priya-r-52433b358 | GitHub: github.com/PriyaR-1719
