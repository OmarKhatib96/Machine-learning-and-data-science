import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('hotel_bookings.csv')
print(dataset.head())

X = dataset.iloc[0:40000, [0,7,8,9,10,11,16,17,18,29]].values
y = dataset.iloc[0:40000, 1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder.fit(X)
X=onehotencoder.transform(X).toarray()
# Avoiding the Dummy Variable Trap
X = X[:, 1:]
#X[:, 0] = labelencoder.fit_transform(X[:, 0])

print(X[0])

#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy=",(100*cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1]),"%")

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, ].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('stays_in_weekends_nights')
plt.ylabel('stays_in_week_nights')
plt.legend()
plt.show()
