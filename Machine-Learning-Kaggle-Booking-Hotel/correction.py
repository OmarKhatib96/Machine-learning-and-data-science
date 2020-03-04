# %% [code]
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# %% [code]
df=pd.read_csv('hotel_bookings.csv')
df.head(10)

# %% [code]
df.tail(10)

# %% [code]
#df['arriavl_date']=pd.to_datetime(df['arrival_date_year']+df['arrival_date_month']+df['arrival_date_day_of_month'])

# %% [code]
print('shape of dataset',df.shape)
print('\n')
print('size of dataset',df.size)

# %% [code]
df.info()

# %% [code]
df.describe().T

# %% [code]
df.describe(include='object').T

# %% [markdown]
# ### EDA

# %% [code]
df.isna().sum()

# %% [code]
cat=df.select_dtypes(include='object').columns
cat

# %% [code]
df=df.drop(['agent','company','reservation_status_date'],axis=1)

# %% [code]
df['country'].mode()

# %% [code]
df['country']=df['country'].replace(np.nan,'PRT')

# %% [code]
df.isnull().sum()

# %% [code]
sns.countplot(df['hotel'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['arrival_date_month'])
plt.show()

# %% [code]

sns.countplot(df['is_canceled'])
plt.show()

# %% [code]
df.is_canceled.value_counts()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['meal'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['market_segment'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['distribution_channel'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['reserved_room_type'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['assigned_room_type'])
plt.show()

# %% [code]

sns.countplot(df['deposit_type'])
plt.show()

# %% [code]


# %% [code]

sns.countplot(df['customer_type'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.countplot(df['reservation_status'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['reservation_status'],df['arrival_date_year'],)
plt.show()

# %% [code]


# %% [code]
df.corr()

# %% [code]
sns.barplot(df['arrival_date_year'],df['previous_cancellations'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_year'],df['previous_bookings_not_canceled'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['previous_cancellations'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['previous_bookings_not_canceled'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_month'],df['is_canceled'])
plt.show()

# %% [code]
plt.figure(figsize=(15 ,10 ))
sns.barplot(df['arrival_date_year'],df['is_canceled'])
plt.show()

# %% [markdown]
# ### converting categorical to numerical

# %% [code]
cat

# %% [code]
df=pd.get_dummies(df,prefix=['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'reserved_room_type', 'assigned_room_type',
       'deposit_type', 'customer_type', 'reservation_status'])

# %% [code]
df.head()

# %% [code]
print('shape of dataset',df.shape)
print('\n')
print('size of dataset',df.size)

# %% [code]
for i in df.columns:
    if (df[i].isnull().sum())!=0:
        print("{} {}".format(i, df[i].isnull().sum()))

# %% [code]
df.children.mode()

# %% [code]
df['children']=df['children'].replace(np.nan,'0')

# %% [code]
df['children']=df['children'].astype('int')

# %% [code]
df.corr()

# %% [code]
plt.figure(figsize=(25 ,20 ))
sns.heatmap(df.corr())

# %% [code]


# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %% [code]
X=df.drop('is_canceled',axis=1 )
y=df['is_canceled']

# %% [code]
LR=LogisticRegression()

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)

# %% [code]
LR.fit(X_train,y_train)

# %% [code]
y_pred = LR.predict(X_test)
y_pred

# %% [code]
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

# %% [code]
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test, y_pred)
accuracy 

# %% [code]


# %% [markdown]
# ### on ols

# %% [code]
import statsmodels.api as sm
X=df.drop('is_canceled',axis=1 )
y=df['is_canceled']

# %% [code]
Xc=sm.add_constant(X)
model=sm.OLS(y,X).fit()
model.summary()

# %% [code]
cols = X.columns.tolist()

while len(cols)>0:
    
    x_1 = X[cols]
    model = sm.OLS(y, x_1).fit()
    p = pd.Series(model.pvalues.values, index = cols)
    pmax = max(p)
    feature_max_p = p.idxmax()
    
    if(pmax > 0.05):
        cols.remove(feature_max_p)
    else:
        break

# %% [code]
print(len(cols))
print(cols)

# %% [code]
X=df[cols]

# %% [code]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=2)

# %% [code]
LR.fit(X_train,y_train)

# %% [code]
y_pred1=LR.predict(X_test)
y_pred1

# %% [code]
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred1)
confusion_matrix

# %% [code]
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test, y_pred1)
accuracy 

# %% [code]


# %% [code]
