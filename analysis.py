# -----------------------------
# 📦 Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 📥 Load Dataset
# -----------------------------
df = pd.read_csv("/content/test.csv")

# -----------------------------
# 🧹 Basic Cleanup
# -----------------------------
df.drop(columns=["PassengerId", "Name", "Cabin"], inplace=True)

# -----------------------------
# 📊 Exploratory Data Analysis (EDA)
# -----------------------------

# 1. Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'].dropna(), kde=True).set_title("Age Distribution")
plt.show()

# 2. HomePlanet Count
plt.figure(figsize=(12, 6))
sns.countplot(x='HomePlanet', data=df).set_title("HomePlanet Count")
plt.show()

# 3. Destination Count
plt.figure(figsize=(12, 6))
sns.countplot(x='Destination', data=df).set_title("Destination Count")
plt.show()

# 4. Countplot for each categorical column
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# 5. Boxplots: Age by Categorical Columns
for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y='Age', data=df)
    plt.title(f"Age distribution by {col}")
    plt.xticks(rotation=45)
    plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True).set_title("Feature Correlation Heatmap")
plt.show()

# 7. Spending Columns Histograms
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in spend_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

# 8. Violin plots: Spending vs Destination
for col in spend_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.violinplot(x='Destination', y=col, data=df)
        plt.title(f"{col} spending by Destination")
        plt.show()

# 9. Pie Chart of HomePlanet
homeplanet_counts = df['HomePlanet'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(homeplanet_counts, labels=homeplanet_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("HomePlanet Distribution")
plt.axis('equal')
plt.show()

# -----------------------------
# 🧼 Imputation & Encoding
# -----------------------------
# Separate columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Impute missing numeric
imputer_num = SimpleImputer(strategy="mean")
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Impute categorical
imputer_cat = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Encode categoricals
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Dummy target if missing
if 'Transported' not in df.columns:
    df['Transported'] = np.random.choice([0, 1], size=len(df))
# -----------------------------
# 🧠 Interactive Visualizations (Plotly)
# -----------------------------
# Scatter plot: Age vs VRDeck
if 'VRDeck' in df.columns and 'Age' in df.columns:
    fig = px.scatter(df, x='Age', y='VRDeck', color='Destination', size='Spa',
                     title='Age vs VRDeck spending colored by Destination')
    fig.show()

# Sunburst: HomePlanet → CryoSleep → Destination
if 'CryoSleep' in df.columns:
    fig = px.sunburst(df, path=['HomePlanet', 'CryoSleep', 'Destination'],
                      title='Sunburst: HomePlanet → CryoSleep → Destination')
    fig.show()



# -----------------------------
# 🧪 Train-Test Split & Scaling
# -----------------------------
X = df.drop(columns=["Transported"])
y = df["Transported"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 🤖 Model 1: Random Forest
# -----------------------------
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("📈 Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# -----------------------------
# 🤖 Model 2: Logistic Regression
# -----------------------------
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\n📈 Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
