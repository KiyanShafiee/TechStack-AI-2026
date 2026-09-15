import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

# 1. Read data with headers
cols = ["rating", 'review_count', 'isbn', 'booktype', 'author_url', 'year', 'genre_urls', 'dir', 'rating_count', 'name']
df = pd.read_csv("data/goodreads.csv", header=None, names=cols)
print(df.head())

# 2. Clean missing year values
df = df[df.year.notnull()]

# 3. Convert data types
df.rating_count = df.rating_count.astype(int)
df.review_count = df.review_count.astype(int)
df.year = df.year.astype(int)

# Fill string NaNs
df.loc[df.genre_urls.isnull(), 'genre_urls'] = ""
df.loc[df.isbn.isnull(), 'isbn'] = ""

# 4. Extract Author Name
def get_author(url):
    name = url.split('/')[-1].split('.')[1:][0]
    return name

df['author'] = df.author_url.map(get_author)

# 5. Extract Genres
def split_and_join_genres(url):
    genres = url.strip().split('|')
    genres = [e.split('/')[-1] for e in genres if e]
    return "|".join(genres)

df['genres'] = df.genre_urls.map(split_and_join_genres)
del df['genre_urls']

# Save cleaned dataframe
df.to_csv("data/cleaned-goodreads.csv", index=False, header=True)

# 6. Grouping: Best book per year
for year, subset in df.groupby('year'):
    bestbook = subset[subset.rating == subset.rating.max()]
    if bestbook.shape[0] > 1:
        print(year, bestbook.name.values, bestbook.rating.values)
    else:
        print(year, bestbook.name.values[0], bestbook.rating.values[0])

# 7. Train / Test Split
X = df[['rating']]
y = df['rating_count']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Training rows:', len(X_train))
print('Test rows:', len(X_test))

# 8. Linear Regression Model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_train = linreg.predict(X_train)
y_pred_test = linreg.predict(X_test)

print("Coefficient (slope):", linreg.coef_[0])
print("Intercept:", linreg.intercept_)
print("Train R^2:", round(r2_score(y_train, y_pred_train), 3))
print("Test R^2:", round(r2_score(y_test, y_pred_test), 3))
