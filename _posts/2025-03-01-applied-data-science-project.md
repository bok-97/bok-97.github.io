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
● information about all beauty products (over 8,000) from the Sephora online store, including product and brand names, prices, ingredients, ratings, and all features.
 ● user reviews (about 1 million on over 2,000 products) of all products from the Skincare category, including user appearances, and review ratings by other users.

## Work Accomplished

### Data Preparation
#### Data Import & Understanding


#### Further Data preparation for Content-Based Recommendation System
1. Create products1 dataframe that includes only product-centric variables
![image](https://github.com/user-attachments/assets/2f3a4001-bda0-4ebf-ab94-d6ff1cdee493)
2. Clean products1 using similar approach (drop_duplicates(), dropna(), reset_index())
![image](https://github.com/user-attachments/assets/f7aa7751-8c26-44bf-8d96-3571e12881e2)
3. Create mapping for product_id  & products1 index to enable fast loopkup by product_id
![image](https://github.com/user-attachments/assets/910aeb3b-4fbe-4bc1-a608-417e340f8657)


#### Further Data preparation for Collaborative Filtering Recommendation System



### Modelling
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

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
