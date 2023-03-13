
###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most important problems in e-commerce is the accurate calculation of post-sale ratings for products.
# Solving this problem means providing greater customer satisfaction for e-commerce sites, highlighting products for
# sellers, and ensuring a seamless shopping experience for buyers. Another problem that arises is the correct sorting of
# product reviews.Misleading reviews can directly affect product sales, leading to both financial loss and customer loss
# By addressing these two fundamental problems, e-commerce sites and sellers can increase their sales while customers
# can complete their purchasing journey seamlessly.

###################################################
# Dataset Story
###################################################

# This dataset containing Amazon product data includes various metadata along with product categories.
# The product with the most reviews in the electronics category has user ratings and reviews.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: rate for helpfullness in review
# reviewText: review
# overall: Product rating
# summary: summary of review
# unixReviewTime: Review Time
# reviewTime: Raw Review Time
# day_diff: Number of days since the review
# helpful_yes: Number of helpful votes for the review
# total_vote: Number of votes given to the review


###################################################
# Task 1: Calculate Average Rating based on current reviews and compare it with the existing Average Rating.
###################################################

# In the shared dataset, users have given ratings and made comments on a product.
# The aim of this task is to evaluate the given ratings by weighting them according to date.
# A comparison is needed between the initial average rating and the weighted rating obtained according to date.


###################################################
# Step 1: Read the dataset and calculate the average rating of the product.
###################################################
import pandas as pd
import numpy as np
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', None)

df = pd.read_csv("amazon_review.csv")

df.head()
df["asin"].nunique()
df.shape


average_rating = df["overall"].mean()


###################################################
# Step 2: Calculate the weighted average rating based on date.
###################################################

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100


time_based_average = time_based_weighted_average(df)
time_based_average > average_rating
# It is expected that the average rating has increased as more weight was given to the recent ratings when considering
# there has been an improvement in rating in the last few days

###################################################
# Step 3: Compare and interpret the average of each time period in weighted rating.
###################################################

# average rating in the last 30 days:
df.loc[df["day_diff"] <= 30, "overall"].mean()
# loc is used since there is an operation on a different column than the condition

# average rating in between last 30 and 90 days:
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()

# average rating in between last 90 and 180 days:
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

# average rating 6 months ago older:
df.loc[(df["day_diff"] > 180), "overall"].mean()

# it seems that overall rating tends to increase over time, but it has experienced a decrease in the last 30 days.


###################################################
# Task 2: Determine the 20 reviews that will be displayed on the product detail page for the product.
###################################################
###################################################
# Step 1. Create helpful_no Variable
###################################################
df.head()
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and add to dataset

df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]

df['score_average_rating'] = np.where(df['total_vote'] == 0, 0, df["helpful_yes"] / df["total_vote"])

# to check: df[["helpful_yes", 'total_vote', 'score_average_rating']][df["total_vote"] == 0]

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB score.
    - The calculated score is used for product ranking.
    -If the scores are between 1-5, they can be transformed into a format suitable for Bernoulli by marking 1-3 negative and 4-5 positive.
    However, this brings some problems with it. Therefore, it is necessary to make a Bayesian average rating."
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

##################################################
# Step 3. Determine 20 reviews and interpret the results
###################################################
df[["reviewText", "wilson_lower_bound"]].sort_values("wilson_lower_bound", ascending=False).head(20)

df.isnull().sum()
