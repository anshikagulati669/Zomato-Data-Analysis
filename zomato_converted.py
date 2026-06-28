
#Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

#Step 2: Load Dataset
df = pd.read_csv("Zomato_with_city_cuisine.csv")

print("Dataset Loaded Successfully")
print(df.head())

#Step 3: Check Column Names
print(df.columns.tolist())

#Step 4: Dataset Understanding
print("Shape of Dataset:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

#Step 5: Data Cleaning
df['rate'] = df['rate'].astype(str)
df['rate'] = df['rate'].str.replace('/5', '', regex=False)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

df.rename(columns={
    'rate':'Rating',
    'votes':'Votes',
    'approx_cost(for two people)':'Cost',
    'city':'City',
    'cuisine':'Cuisine'
}, inplace=True)
df['Rating'] = df['Rating'].fillna(df['Rating'].median())

print(df[['Rating']].head())

#Step 6: Verify Cleaning
print(df.dtypes)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

#Step 7: Descriptive Statistics
numerical_cols = ['Votes', 'Cost', 'Rating']

for col in numerical_cols:

    print("\n", "="*40)

    print("Column:", col)

    print("Mean:", df[col].mean())

    print("Median:", df[col].median())

    print("Mode:", df[col].mode()[0])

    print("Standard Deviation:", df[col].std())

    print("Variance:", df[col].var())

#Step 8: Restaurant Distribution Analysis
#Highest Restaurant City
city_count = df['City'].value_counts()

print("City with Highest Restaurants:")

print(city_count.idxmax())

print("Count:", city_count.max())
#Top 10 Cities
top10 = df['City'].value_counts().head(10)

print(top10)
#Visualization
plt.figure(figsize=(10,5))

sns.barplot(
    x=top10.index,
    y=top10.values
)

plt.title("Top 10 Cities by Restaurant Count")

plt.xlabel("City")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.show()

#Task 5: Cuisine Analysis
top_cuisines = df['Cuisine'].value_counts().head(10)

print(top_cuisines)
plt.figure(figsize=(10,6))

sns.countplot(
    y='Cuisine',
    data=df,
    order=df['Cuisine'].value_counts().index[:10]
)

plt.title("Top 10 Most Common Cuisines")

plt.show()

#Q2 Which Cuisines Have Highest Ratings?
highest_rated = (
    df.groupby('Cuisine')['Rating']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print(highest_rated)
plt.figure(figsize=(10,6))

sns.barplot(
    x=highest_rated.values,
    y=highest_rated.index
)

plt.title("Top Rated Cuisines")

plt.xlabel("Average Rating")

plt.ylabel("Cuisine")

plt.show()

#Task 6: Rating Analysis
#Q1 Distribution of Ratings
plt.figure(figsize=(8,5))

sns.histplot(
    df['Rating'],
    bins=10,
    kde=True
)

plt.title("Rating Distribution")

plt.show()

#Q2 Calculate Skewness
print("Skewness =", df['Rating'].skew())
if df['Rating'].skew() > 0:
    print("Positively Skewed")
elif df['Rating'].skew() < 0:
    print("Negatively Skewed")
else:
    print("Symmetric Distribution")

#Q3 Detect Outliers Using IQR
Q1 = df['Rating'].quantile(0.25)

Q3 = df['Rating'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df['Rating'] < lower_limit) |
    (df['Rating'] > upper_limit)
]

print("Number of Outliers =", len(outliers))

print(outliers)

#Q4 Boxplot
plt.figure(figsize=(8,5))

sns.boxplot(x=df['Rating'])

plt.title("Rating Boxplot")

plt.show()

#Task 7: Online Delivery Analysis
#Q1 Restaurants Providing Online Delivery
delivery_count = df['online_order'].value_counts()

print(delivery_count)
#Q2 Compare Average Ratings
avg_rating = (
    df.groupby('online_order')['Rating']
    .mean()
)

print(avg_rating)
#Q3 Visualization
plt.figure(figsize=(6,4))

sns.barplot(
    x=avg_rating.index,
    y=avg_rating.values
)

plt.title("Average Rating by Online Delivery")

plt.xlabel("Online Delivery")

plt.ylabel("Average Rating")

plt.show()

#Task 8: Cost Analysis
#Q1 Average Restaurant Cost
print("Average Cost = ₹", round(df['Cost'].mean(),2))
max_cost = df['Cost'].max()
#Q2 Most Expensive Restaurants
expensive_restaurants = df[
    df['Cost'] == max_cost
]

print(
    expensive_restaurants[
        ['name','Cost','Rating']
    ]
)
#Q3 Relationship Between Cost and Rating
correlation = df['Cost'].corr(df['Rating'])

print("Correlation =", correlation)
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='Cost',
    y='Rating',
    data=df
)

plt.title("Cost vs Rating")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(x=df['Cost'])

plt.title("Cost Distribution")

plt.show()

#Task 9: Correlation Analysis
#Q1 Correlation Matrix
corr_matrix = df[
    ['Votes','Cost','Rating']
].corr()

print(corr_matrix)
#Q2 Heatmap'
plt.figure(figsize=(7,5))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()
#Q3 Strongest Relationship
corr_matrix

#Task 10: Feature Engineering
#Q1 Create Cost Category
def cost_category(cost):

    if cost <= 300:
        return "Budget"

    elif cost <= 600:
        return "Moderate"

    elif cost <= 1000:
        return "Premium"

    else:
        return "Luxury"

df['Cost_Category'] = df['Cost'].apply(
    cost_category
)

print(df['Cost_Category'].value_counts())
plt.figure(figsize=(6,4))

sns.countplot(
    x='Cost_Category',
    data=df
)

plt.title("Cost Category Distribution")

plt.show()

#Q2 Create Rating Category
def rating_category(rating):

    if rating <= 3:
        return "Poor"

    elif rating <= 3.5:
        return "Average"

    elif rating <= 4:
        return "Good"

    else:
        return "Excellent"

df['Rating_Category'] = df['Rating'].apply(
    rating_category
)

print(df['Rating_Category'].value_counts())
plt.figure(figsize=(6,4))

sns.countplot(
    x='Rating_Category',
    data=df
)

plt.title("Rating Category Distribution")

plt.show()
pd.crosstab(
    df['Cost_Category'],
    df['Rating_Category']
)

#
