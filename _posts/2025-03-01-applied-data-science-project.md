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

### About Dataset
This work utilizes the Sephora Products and Skincare Reviews datasets provided by Nady Inky on Kaggle, it contains:
- information about all beauty products (over 8,000) from the Sephora online store, including product and brand names, prices, ingredients, ratings, and all features.
- user reviews (about 1 million on over 2,000 products) of all products from the Skincare category, including user appearances, and review ratings by other users.

## Work Accomplished

### Data Preparation
#### Data Import & Understanding


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



### Modelling
<img width="190" alt="image" src="https://github.com/user-attachments/assets/bcc42eea-9849-42ce-a897-81995d9e0ace" />

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

### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Github repo: https://github.com/bok-97/itd214_project_data
Datasource (kaggle): https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data
