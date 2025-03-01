---
layout: post
author: Chia Hoi BOK
title: "Product Recommendation System for Sephora Online Store"
categories: ITD214
---
## Project Background
This work is a part of a group project module from Nanyang Polytechnic Singapore's *Specialist Diploma in Business & Big Data Analytics*.

#### Project Details
- Project module  : ITD214 - Applied Data Science Project
- Group 5         : Applying Data Science to Improve/Optimize Skincare Business Model Through Customer Reviews
- Business Goal   : To enchance customer experience and business profitability by utilizing customer insights, product performance data, and market trends to drive product improvements, optimize marketing 
strategies, and refine pricing decisions.

This workbook is Part 3 of of the group project, titled **Product Recommendation System Development**, with the objective of developing a recommendation system to suggest products based on product similarity, customer purchasing patterns and reviews for a Sephora online store.

![image](https://github.com/user-attachments/assets/b09ad82b-ccdc-4bb6-8f97-c261de77efb9)

A recommendation system is an AI-driven tool that analyzes user behavior and preferences to suggest relevant products, services, or content. It uses techniques like collaborative filtering, content-based filtering, and hybrid approaches to provide personalized recommendations. These systems are widely used in e-commerce, streaming platforms, and online services to enhance user experience and engagement.

In this work, we will be developing two types of recommendation systems:
1. Content-Based Recommender System; recommend based on product similarity/cosine-similarity
2. Collaborative Filtering Recommender System; recommend based on user preference/rating hisotry

<img width="190" alt="image" src="https://github.com/user-attachments/assets/bcc42eea-9849-42ce-a897-81995d9e0ace" />

### About Dataset
This work utilizes the Sephora Products and Skincare Reviews datasets provided by Nady Inky on Kaggle, it contains:
- information about all beauty products (over 8,000) from the Sephora online store, including product and brand names, prices, ingredients, ratings, and all features.
- user reviews (about 1 million on over 2,000 products) of all products from the Skincare category, including user appearances, and review ratings by other users.

1. Product Information in product_info.csv file
- product_id: The unique identifier for each product.
- product_name: The name of the product.
- brand_id: The unique identifier for each brand.
- brand_name: The name of the brand.
- loves_count: The number of "loves" each product has received from users.
- rating: The average rating for the product.
- reviews: The number of reviews the product has.
- size: The size of the product in oz/ml.
- variation_type: The type of variation, if any, for the product.
- variation_value: The value of the variation, if any, for the product.
- variation_desc: The description of the variation, if any, for the product.
- ingredients: The ingredients of the product.
- price_usd: The price of the product in USD.
- value_price_usd: The value price of the product in USD.
- sale_price_usd: The sale price of the product in USD.
- limited_edition: Boolean value indicating whether the product is a limited edition.
- new: Boolean value indicating whether the product is new.
- online_only: Boolean value indicating whether the product is available online only.
- out_of_stock: Boolean value indicating whether the product is out of stock.
- sephora_exclusive: Boolean value indicating whether the product is exclusive to Sephora.
- highlights: The highlights of the product.
- primary_category: The primary category of the product.
- secondary_category: The secondary category of the product.
- tertiary_category: The tertiary category of the product.
- child_count: The number of child products, if any.
- child_max_price: The maximum price among child products, if any.
- child_min_price: The minimum price among child products, if any.

2. Customer Reviews in reviews_0_250.csv to reviews_1500_end.csv files
- author_id: Unique identifier for each author(user).
- rating: The rating given by the user for that product.
- is_recommended: Boolean value indicating whether the user would recommend the product.
- helpfulness: Indicator of how helpful other users found the review.
- total_feedback_count: The total feedback count for the review.
- total_neg_feedback_count: The total count of negative feedback for the review.
- total_pos_feedback_count: The total count of positive feedback for the review.
- submission_time: The date the review was submitted.
- review_text: The text of the review.
- review_title: The title of the review.
- skin_tone: The skin tone of the user.
- eye_color: The eye color of the user.
- skin_type: The skin type of the user.
- hair_color: The hair color of the user.
- product_id: Unique identifier for each product.
- product_name: The name of the product.
- brand_name: The name of the brand.
- price_usd: The price of the product in USD when the review was written.

## Work Accomplished

### Data Preparation and Cleaning
#### Data Import & Understanding
1. Importing necessary libraries
````html
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from surprise import Dataset, Reader, SVD
#from surprise.model_selection import cross_validate
````
2. Importing datafiles as product and review, checking .info() and head() for data understanding
````html
# Load product datafile
product = pd.read_csv("product_info.csv")
product.info()
product.head()
# renaming rating column
product = product.rename(columns={'rating': 'ave_rating'})

# Load review datafiles and concatenate
df1 = pd.read_csv("reviews_0-250.csv", low_memory=False)
df2 = pd.read_csv("reviews_250-500.csv", low_memory=False)
df3 = pd.read_csv("reviews_500-750.csv", low_memory=False)
df4 = pd.read_csv("reviews_750-1250.csv", low_memory=False)
df5 = pd.read_csv("reviews_1250-end.csv", low_memory=False)
review = pd.concat([df1,df2,df3,df4,df5])
review.info()
review.head()
````
3. Checking for data summary
````html
def data_summary(df, name):
    print(f"\033[1m\n{name} Data Summary\033[0m")
    print("Rows     :", df.shape[0])
    print("Columns  :", df.shape[1])
    print("\nFeatures :\n", df.columns.tolist())
    print("\nMissing values : ", df.isnull().sum().values.sum())
    print("\nUnique values :\n", df.nunique())

data_summary(product, "Product")
data_summary(review, "Review")
````
4. Selecting only necessary columns for recommendation systems
````html
# Select only necessary columns for recommendation system
product_df = product[['product_id', 'product_name', 'brand_name', 'primary_category', 'secondary_category', 'tertiary_category','highlights']]
review_df = review[['author_id', 'product_id', 'rating', 'review_text', 'skin_tone', 'eye_color', 'skin_type', 'hair_color']]
````
5. Merging datasets by key column 'product_id'
````
# Merge datasets on 'product_id'
data = review_df.merge(product_df, on='product_id', how='left')
data.head().transpose()
````
6. Dropping columns with excessive missing values (more than 80%)
````
data1 = data.dropna(thresh=len(data) * 0.80, axis=1)
````
7. Filling missing 'rating' values by product mean
```
data['rating'] = data.groupby('product_id')['rating'].transform(lambda x: x.fillna(x.mean()))
```
8. Dropping missing values from other columns (<20% of total values)
````
data = data.dropna()
````
9. Dropping duplicates from all columns
````
data = data.drop_duplicates()
````
10. Resetting the index of the DataFrame to overwrite old index
````
data = data.reset_index(drop=True)
````

#### Basic Exploratory Data Analysis (EDA) of data dataframe
1. Checking simple statistics of numeric variables (ie. ratings). Rating : This is can be considered as categorical attribute with values of 1,2,3,4,5. The mean rating is 4.3 which means most of the users have given very good ratings (>4).
````
data.describe().round(2).transpose()
````
2. Data (distribution) visualisation through ipywidget (dropdown)
````
column_dropdown = widgets.Dropdown(
    options=['rating', 'skin_tone', 'eye_color', 'skin_type', 'hair_color', 
             'primary_category', 'secondary_category', 'tertiary_category'],
    value='rating',  # Default selection
    description='Feature:',
    style={'description_width': 'initial'}
)

# Function to plot the selected feature distribution
def plot_distribution(feature):
    plt.figure(figsize=(10, 5))
    
    if data[feature].dtype == 'object':  # Categorical Features
        counts = data[feature].value_counts()
        plt.barh(counts.index, counts.values, color='skyblue')
        plt.xlabel("Count")
        plt.ylabel(feature)
        plt.title(f"Distribution of {feature}")
    else:  # Numeric Feature (ave_rating)
        plt.hist(data[feature], bins=10, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature}")

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Interactive output
widgets.interactive(plot_distribution, feature=column_dropdown)
````
Sample output of 'rating' dropdown selection:

![image](https://github.com/user-attachments/assets/789a3088-8924-4c15-9072-d23e56907356)


#### Further Data preparation for Content-Based Recommendation System
A content-based recommendation system recommends items to the users that are relevant to the preferred features of other items.

For example, if a user often searches or browses for ‘black dress’ on a shopping e-platform, a content-based recommendation system will recommend the user other dresses of the same colour.
In this section we will compute the pairwise similarity score of all products based on their features and recommend products accordingly. The features are in the highlights column of our dataset.

The similarity between two products can be calculated by measuring the cosine of the angle between two vectors in a matrix. To calculate it, the text strings (our highlights lists) is converted to word vectors in a matrix. Then, the angle between vectors is calculated and a score from 0 to 1 is generated. Values closer to 0 show low similarity and values closer to 1 show high similarity.

1. Create products1 dataframe that includes only product-centric variables
```html
products1 = pd.DataFrame(product, columns=[
    'product_id','product_name','highlights','primary_category', 'secondary_category', 'tertiary_category'])
products1.head(5)
products1.shape
````
2. Clean products1 using similar approach (drop_duplicates(), dropna(), reset_index())
```html
products1 = products1.drop_duplicates()
products1 = products1.dropna()
products1 = products1.reset_index(drop=True)
products1.head()
products1.isnull().sum()
````
3. Create mapping for product_id  & products1 index to enable fast loopkup by product_id
````html
# Constructing a series from a dictionary with data indices and index product_name
indices = pd.Series(products1.index, index=products1['product_id'])
````
4. Extract highlights and categories and convert to string
````html
# Extract highlights and categories and convert to string
texts = (
    products1['highlights'].astype(str) + " " +
    products1['primary_category'].astype(str) + " " +
    products1['secondary_category'].astype(str) + " " +
    products1['tertiary_category'].astype(str)
).values
texts
````
5. Vectorize by TF-IDF for highlights and categories
````html
# TF-IDF Vectorizer for highlights and categories
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(texts)
tfidf_matrix.shape

# Convert the TF-IDF matrix to a dense format and then into a DataFrame
tfidf_dense = tfidf_matrix.toarray()

# Create a DataFrame to display the TF-IDF values
tfidf_df = pd.DataFrame(tfidf_dense, columns=tfidf.get_feature_names_out())

# Display the first few rows
tfidf_df.head()
# 285 different words were used to describe 5484 products.
````
6. Compute cosine-similarity based on product
````html
# Content-based similarity using cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
````

#### Further Data preparation for Collaborative Filtering Recommendation System
A collaborative recommendation system predicts user preferences by leveraging past interactions and preferences of similar users, typically through collaborative filtering techniques such as user-based or item-based approaches. These systems focus on identifying patterns in user behavior to suggest items, relying on the assumption that users who agreed in the past will agree in the future (Ricci, F., Rokach, L., & Shapira, B., 2015).

Example: If User A and User B have rated many movies similarly, and User B gives a high rating to a movie that User A hasn’t seen, we can recommend that movie to User A. Since User B rated Movie 3 as 5, we can predict that User A might like Movie 3 as well.
User              | Movie 1           | Movie 2           | Movie 3          | Movie 4
----------------- | ----------------- | ----------------- | -----------------| -----------------
A                 | 5                 | 4                 | **?**            | 2
B                 | 5                 | 4                 | 5                | 3

Collaborative filtering systems work in 5 simple steps:
1. Collect User-Item Interactions (Load Data) – Gather data on user behavior, such as ratings, purchases, or clicks on items.
2. Build a User-Item Matrix – Create a matrix where rows represent users, columns represent items, and values show interactions (e.g., ratings).
3. Find Similarities – Identify similar users (User-Based CF) or similar items (Item-Based CF) based on their interaction patterns.
4. Predict Ratings/Preferences – Estimate a user’s interest in an item by analyzing ratings from similar users or items.
5. Recommend Top Items – Suggest the highest predicted items that the user hasn’t interacted with yet.

Making Predictions using Surprise library:
Surprise is a popular library for building and analyzing recommendation systems, as it provides various ready-to-use algorithms and tools to evaluate and compare the performance of these algorithms. Surprise automates similarity calculations and matrix factorization (e.g., SVD), making it more efficient than manually computing similarities!

1. Importing important libraries
````html
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
````
To load a surprise dataset from a pandas dataframe, we will use the load_from_df() method, we will also need a Reader object, and the rating_scale parameter must be specified. The dataframe must have three columns, corresponding to the **user ids, the item ids, and the ratings** in this order. Each row thus corresponds to a given rating.
2. Removing products with low review counts (20% percentile)
````html
product_threshold = data['product_id'].value_counts().quantile(0.2)

# Creating cut-off filter for less-reviewed products
filtered_products = data['product_id'].value_counts()

# Filters products with review counts > product_threshold (34 reviews)
filtered_products = filtered_products[filtered_products > product_threshold].index
print(filtered_products)
# filtered_products filter will reduce product count from 1627 to 1297.
````
3. Removing authors with low review counts (80% percentile)
````html
author_threshold = data['author_id'].value_counts().quantile(0.8)

# Creating cut-off filter for authors with less than 2 reviews
filtered_authors = data['author_id'].value_counts()

# Filtering authors with review counts > product_threshold (2)
filtered_authors = filtered_authors[filtered_authors > author_threshold].index
# filtered_authors filter will reduce author count from 324222 to 59893.
````
4. Apply the filtered_authors & filtered_products filters
````html
filtered_data = data[(data['product_id'].isin(filtered_products)) & 
                     (data['author_id'].isin(filtered_authors))]
````
To load a surprise dataset from a pandas dataframe, we will use the load_from_df() method, we will also need a Reader object, and the rating_scale parameter must be specified.
The dataframe must have three columns, corresponding to the author_id, the product_id, and the ratings in this order.
5. Load the surprise dataset format
````html
# Surprise: Specifiess the minimum (1) and maximum (5) possible rating values in the dataset
reader = Reader(rating_scale=(1,5))

# Surprise: Loading from pandas dataframe to surpise dataset
surprise_data = Dataset.load_from_df(filtered_data[['author_id', 'product_id', 'rating']], reader)
````
6. Split into train and test sets (80:20 ratio)
````html
trainset, testset = train_test_split(surprise_data, test_size=0.2,random_state=5)
````

### Modelling

#### 1. Content-based Recommendation System
A function get_recommendation() is created to call the top 5 similar products to the user-input product_id.
````html
# Recommendation Function
def get_recommendations(product_id, cosine_sim=cosine_sim):
    if product_id not in indices:
        return f"Product ID {product_id} not found."
    idx = indices[product_id]
    product_name = products1.loc[products1['product_id'] == product_id, 'product_name'].values[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar products
    product_indices = [i[0] for i in sim_scores]
    recommendations = products1.iloc[product_indices][['product_id', 'product_name']]
    print(f"Product Selected: {product_name} (ID: {product_id})\n")
    print("Top 5 Recommended Products:")
    
    return recommendations

# Prompt user to input a product code
user_product_id = input("Please enter the product code: ")

# Call the recommendation function with user input
print(get_recommendations(user_product_id))
````
Example of inputting a fragrance product (P473671), the top 5 recommendations are also fragrance products:

<img width="323" alt="image" src="https://github.com/user-attachments/assets/51971efc-c46b-4d20-921c-de1859722cc7" />

#### 2. Collaborative Filtering Recommendation System

##### 2.1 SVD Matrix Factorization Algorithm
Singular Value Decomposition (SVD):
- A matrix factorization technique that decomposes a given matrix into a set of matrices.
- Reduces the dimensionality of user-item interaction matrices and generates latent factors for both users and items that can then be used to predict user preferences for items and provide personalized recommendations.
- SVD is known for its accuracy in collaborative filtering-based recommendation systems and has been popularized by its use in the Netflix Prize competition.

1. Train a Collaborative Filtering Model (SVD, Untuned)
````html
model = SVD()
model.fit(trainset)
````
2. Make Predictions on the Test Set
````html
predictions = model.test(testset)
````
The aobve returns a list of prediction objects used by the surprise library to store results:
- uid – (inner) user id.
- iid – (inner) item id
- r_ui (float) - the true rating r_ui
- est (float) The estimated rating r_ui
- details (dict) – Stores additional details about the prediction that might be useful for later analsis.

### Evaluation 
Evaluation of untuned SVD model:
````html
# Evaluate model accuracy by built-in RMSE Metric in SURPRISE
mse = accuracy.mse(predictions)
rmse = accuracy.rmse(predictions)
````
<img width="486" alt="image" src="https://github.com/user-attachments/assets/6cb7c643-ad01-4257-b316-d1204a747281" />

We can use a more rigorous cross_validation evaluation on the untuned SVD model.
```html
def validate_model(model, data):
    results = cross_validate(model, data, measures=['RMSE', 'MAE', 'FCP'], cv=10, verbose=True) # cross-validate 10 folds
    return pd.DataFrame.from_dict(results).mean(axis=0)

svd_1 = validate_model(model, surprise_data)
print(svd_1)
````
<img width="559" alt="image" src="https://github.com/user-attachments/assets/d2cbbd55-37e4-46a6-8698-9df1c4d2dd6f" />

1. RMSE (Root Mean Square Error): Measures how far the predicted ratings are from actual ratings. Lower RMSE is better (closer to zero means better accuracy).
- RMSE = 0.9192 (model's average predicted ratings are about 0.95 stars away from actual ratings)

2. MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual ratings. Similary, lower is better. MAE does not penalize large errors as much as RMSE does.
- Mean MAE = 0.6598 (predicted ratings are on average off by 0.66 stars)

3. FCP (Fraction of Concordant Pairs): Measures how well the model ranks items. Higher FCP is better (closer to 1 means better ranking).
- Mean FCP = 0.5140 (in 51% of cases, the model correctly ranked higher-rated items above lower-rated ones). FCP is slightly better than random (0.5), but there's room for improvement.

#### Untuned SVD Model Application


### SVD Model Hyperparameter Tuning
We will now attempt to do hyperparameter tuning for the SVD model to find out the optimal parameter values for the model. We will be comparing the performance between GridSearchCV and RandomizedSearchCV methods and select one that will produce lower RMSE.
1. Importing built-in surprise functions
````html
from surprise import accuracy
from surprise.model_selection.validation import cross_validate
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise.model_selection import RandomizedSearchCV

reader = Reader()
surprise_data = Dataset.load_from_df(filtered_data[['author_id', 'product_id', 'rating']], reader)
````
2. Defining parameter grid for GridSearchCV and RandomizedSearchCV
````html
# Define parameter grid for GridSearchCV and RandomizedSearchCV
param_grid = {
    'n_factors': [10, 20, 50, 100], # number of latent features to represent users and items in a lower-dimensional space
    'n_epochs': [5, 10, 20], # number of iterations for training
    'lr_all': [0.002, 0.005, 0.01], # learning rate for all parameters, how much model updates weights/iteration
    'reg_all': [0.02, 0.1, 0.3]  # regularization strength to prevent overfitting, reduces model complexity but may cause underfitting
}
````
3. Fitting both GridSearchCV and RandomizedSearchCV to dataset
````html
# GridSearchCV: Exhaustive search for best parameters
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(surprise_data)

# RandomizedSearchCV: Random sampling from parameter grid
random_search = RandomizedSearchCV(SVD, param_grid, n_iter=10, measures=['rmse'], cv=3, random_state=42)
random_search.fit(surprise_data)
````
4. Calculate RMSE for both searches, compare and outputs the tuned hyperparameters
````html
# Get RMSE from both searches
grid_rmse = grid_search.best_score['rmse']
random_rmse = random_search.best_score['rmse']

# Compare RMSEs and select the best model
if grid_rmse < random_rmse:
    print(f"GridSearchCV gives the best model with RMSE: {grid_rmse}")
    best_model = grid_search.best_estimator['rmse']
    best_params = grid_search.best_params
else:
    print(f"RandomizedSearchCV gives the best model with RMSE: {random_rmse}")
    best_model = random_search.best_estimator['rmse']
    best_params = random_search.best_params

# Output the tuned hyperparameters
print("Tuned hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
````
The output from the above is:

<img width="353" alt="image" src="https://github.com/user-attachments/assets/bd96c2f3-087d-4b8c-b202-30cb4b3b8018" />

5. Fine-tuning SVD model with obtained parameter values and fitting it
````html
final_model = SVD(n_factors=100, 
                  n_epochs=20, 
                  lr_all=0.01, 
                  reg_all=0.1)

# Fit the final model on the entire trainset
final_model.fit(trainset)
````
6. Evaluating the tuned-SVD model using cross validation
````html
def validate_model_after_training(model, data):
    """Validate the trained model with cross-validation and return the results."""
    results = cross_validate(model, data, measures=['RMSE', 'MAE', 'FCP'], cv=10, verbose=True)  # 10-fold cross-validation
    return pd.DataFrame.from_dict(results).mean(axis=0)

# Validate the trained model
print("Validating the trained model with cross-validation...")
cv_results = validate_model_after_training(final_model, surprise_data)

print("Cross-validation results after training:")
print(cv_results)

# **Final Evaluation on the Testset**
predictions = final_model.test(testset)
final_rmse = accuracy.rmse(predictions)
print(f"Final RMSE after retraining: {final_rmse}")
````
Output:

<img width="557" alt="image" src="https://github.com/user-attachments/assets/b1562bda-d9df-4aca-932e-7b4b772adc54" />

#### Tuned-SVD Model Application
A new function is defined to get the top 5 recommendation products, and an example output is shown using author_id '5182718480'.
````html
# Function to Recommend Top-N Products for a User; N = 5
def get_top_n_recommendations(model, user_id, n=5):
    """Recommend top N products for a given user."""
    # Get all product IDs
    all_product_ids = filtered_data['product_id'].unique()
    
    # Remove products user has already rated
    rated_products = filtered_data[filtered_data['author_id'] == user_id]['product_id'].values
    unrated_products = [pid for pid in all_product_ids if pid not in rated_products]
    
    # Predict ratings for unrated products
    predictions = [final_model.predict(user_id, pid) for pid in unrated_products]
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-N recommendations
    top_n = predictions[:n]

    # Convert to DataFrame for table output
    df = pd.DataFrame([(pred.iid, pred.est) for pred in top_n], columns=["Product ID", "Predicted Rating"])
    df.index += 1  # Start index from 1 for better readability
    
    return [(pred.iid, pred.est) for pred in top_n]

# Take user input for user_id
user_id = input("Enter User ID: ")


try:
    user_id = int(user_id)  # Convert input to integer if needed
    recommendations_df = get_top_n_recommendations(model, user_id, n=5)
    print(recommendations_df)  # Display recommendations in table format
except ValueError:
    print("Invalid User ID. Please enter a numeric value.")
````

<img width="571" alt="image" src="https://github.com/user-attachments/assets/f628350c-4d0a-4401-962c-0469a059eb2f" />

We find out that, for the same author_id, the same top 5 products were recommended. Plotting a side-by-side bar chart of the metric scores comparison between the untuned and tuned SVD model, it explains why both models are giving the same outcome.
````html
import pandas as pd
import matplotlib.pyplot as plt
results_df = pd.DataFrame({
    'SVD without tuning': [svd_1['test_rmse'], svd_1['test_mae'], svd_1['test_fcp']],
    'SVD after tuning': [cv_results['test_rmse'], cv_results['test_mae'], cv_results['test_fcp']]
}, index=['RMSE', 'MAE', 'FCP'])

print("Comparison DataFrame:")
print(results_df)

# Plotting the comparison chart (side-by-side bar chart)
ax = results_df.plot(kind='bar', figsize=(10, 6))
ax.set_title('Comparison of SVD Models (Tuned vs. Untuned)')
ax.set_ylabel('Metric Value')
ax.set_xlabel('Evaluation Metrics')
plt.xticks(rotation=0)
plt.legend(title='Model')
plt.tight_layout()
plt.show()
````

<img width="573" alt="image" src="https://github.com/user-attachments/assets/79680647-cb30-4ab6-82cf-b1a46f7f33e0" />

#### Other Built-in Algorithms in SURPRISE Library
The surprise library also offers other algorithms that supports recommendation system models. A few of them are explored and compared with in terms of RMSE performance here.

1. Importing other algorithms and defining function to iterate cross validation on the dataset  
````html
from surprise import NMF
from surprise import SVDpp
from surprise import KNNWithZScore
from surprise import BaselineOnly
from surprise import CoClustering

benchmark = []
# Iterate over all algorithms
for algorithm in [NMF(), SVD(), SVDpp(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, surprise_data, measures=['RMSE'], cv=5, verbose=False, n_jobs=1)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = pd.concat([tmp, pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])])
    benchmark.append(tmp)
    
comparison = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse', ascending=True)
````
2. Plotting the comparison results
````html
# Set the figure size
fig, ax = plt.subplots(figsize=(12, 6))

# Set up the position of the bars
bar_width = 0.25
index = np.arange(len(comparison))

# Plot the bars for each metric
bar1 = ax.bar(index - bar_width, comparison['test_rmse'], bar_width, label='test_rmse', color='skyblue')
bar2 = ax.bar(index, comparison['fit_time'], bar_width, label='fit_time', color='orange')
bar3 = ax.bar(index + bar_width, comparison['test_time'], bar_width, label='test_time', color='lightgreen')

# Set labels and title
ax.set_xlabel('Algorithm')
ax.set_ylabel('Values')
ax.set_title('Comparison of Algorithms by Metrics')
ax.set_xticks(index)
ax.set_xticklabels(comparison.index, rotation=45)

# Add a legend
ax.legend()

# Tight layout for better display
plt.tight_layout()

# Show the plot
plt.show()
````

Outcome is shown below. It can be seen that SVDpp and SVD models have the least RMSE values and is the optimal model used for this work.

<img width="529" alt="image" src="https://github.com/user-attachments/assets/8fef16a8-fcac-44d1-8e82-18690f9375c3" />


## Recommendation and Analysis

Recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries). Recommender systems are really critical in some industries as they can generate a huge amount of income when they are efficient or also be a way to stand out significantly from competitors.

We have achieved the business objective of building a recommendation system for users of Sephora products.
We have used 2 different types here:

1. Content-Based Recommender System; recommend based on product similarity/cosine-similarity
2. Collaborative Filtering Recommender System; recommend based on user preference/rating hisotry
The best model used recommended is **SVD-based** algorithm where it had acheived the lowest RMSE when compared to other SURPRISE algorithms (NMF, CoClustering, BaselineOnly).

Model optimzation was also attempted by **tuning the hyperparameters of SVD through GridSearchCV** before applying fitting the optimized parameters (epochs, learning rates, etc) to the dataset for training and testing.

### Recommendation for Further Optimization
Surprise models rely on the training data. If a user was removed before training, the model never saw them, so it does not have their preferences and no past ratings to base predictions on. This is known as Cold Start Problem (New or Unseen Users).

1. Keep Low-Activity Users in Training Instead of removing them completely, keep them but apply different weights (e.g., give more weight to active users).
2. Use a "Default" or Hybrid Approach If a user wasn’t in training, fall back on item popularity-based recommendations.
3. Implement a Different Model for Cold Start Use content-based filtering (e.g., recommend items similar to what they view) until enough reviews exist.

<img width="307" alt="image" src="https://github.com/user-attachments/assets/ac1bc278-0099-4e18-945a-86ea8c134687" />

#### Hybrid Model (Combining Collaborative Filtering Model with Content-Based System)
````html
# Compute Product Similarity Based on Features
feature_columns = ['skin_tone', 'eye_color', 'skin_type', 'hair_color']
df_encoded = pd.get_dummies(filtered_data[feature_columns])  # One-hot encode categorical features

product_features = df_encoded.groupby(filtered_data['product_id']).mean()  # Aggregate features per product
product_similarity = cosine_similarity(product_features)  # Compute cosine similarity

# Convert to DataFrame for easy lookup
product_sim_df = pd.DataFrame(product_similarity, index=product_features.index, columns=product_features.index)

# Function to Recommend Top-N Products (Hybrid)
def get_hybrid_recommendations(user_id, n=5):
    """Hybrid Recommendation: Collaborative Filtering + Content Features"""
    
    # Get all unique product IDs
    all_product_ids = filtered_data['product_id'].unique()
    
    # Get products the user has already rated
    rated_products = filtered_data[filtered_data['author_id'] == user_id]['product_id'].values
    unrated_products = [pid for pid in all_product_ids if pid not in rated_products]
    
    # Predict ratings for unrated products using SVD
    predictions = [model.predict(user_id, pid) for pid in unrated_products]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Select top-N recommended products
    top_n_products = [pred.iid for pred in predictions[:n]]
    
    # Enhance with feature-based similarity
    hybrid_scores = []
    for pid in top_n_products:
        similar_products = product_sim_df[pid].sort_values(ascending=False).index[1:3]  # Get 2 most similar products
        hybrid_scores.extend(similar_products)
    
    # Return final recommendations
    final_recommendations = list(set(top_n_products + hybrid_scores))[:n]
    
    return final_recommendations

# Example Usage: Recommend top 5 products for a user
user_id = 5442418082  # Example user
print("Top 5 Hybrid Recommendations:", get_hybrid_recommendations(user_id, n=5))
````

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Datasource files are too large to upload. 
Datasource (kaggle): https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data

Github repo: https://github.com/bok-97/itd214_project_data

