

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import adam
import tensorflow as tf
from os import listdir
from os.path import join
from csv import DictReader

# Importing the dataset
df = pd.read_csv('hotel_bookings.csv')

cat=df.select_dtypes(include='object').columns

df=df.drop(['agent','company','reservation_status_date'],axis=1)

df['country'].mode()

df['country']=df['country'].replace(np.nan,'PRT')

# Fitting the ANN to the Training set
df=pd.get_dummies(df,prefix=['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'reserved_room_type', 'assigned_room_type',
       'deposit_type', 'customer_type', 'reservation_status'])

df.head()

print('shape of dataset',df.shape)
print('\n')
print('size of dataset',df.size)

for i in df.columns:
    if (df[i].isnull().sum())!=0:
        print("{} {}".format(i, df[i].isnull().sum()))

df.children.mode()

df['children']=df['children'].replace(np.nan,'0')

df['children']=df['children'].astype('int')

df.corr()

plt.figure(figsize=(25 ,20 ))



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X=df.drop('is_canceled',axis=1 )
y=df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)




# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 130, kernel_initializer = 'uniform', activation = 'relu', input_dim = 258))
# Adding the second hidden layer
classifier.add(Dense(units = 130, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
opt = adam(lr=0.0004, decay=1e-6)

# Compiling the ANN
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
csv_logger =tf.keras.callbacks.CSVLogger('log.csv')
 

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 15, epochs = 15,callbacks=[csv_logger])


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy=",100*(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1]),"%")



def plot_log(filename, show=None):
    """ Plot log of training / validation learning curve
    # Arguments
        :param filename: str, csv log file name
        :param show: None / str, show graph if none or save to 'show' directory
    """
    # Load csv file
    keys, values, idx = [], [], None
    with open(filename, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            if len(keys) == 0:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                idx = keys.index('epoch')
                continue
            for _, value in row.items():
                values.append(float(value))
        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:, idx] += 1
    # Plot
    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        # training loss
        if key.find('loss') >= 0:   # and not key.find('val') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')
    fig.add_subplot(212)
    for i, key in enumerate(keys):
        # acc
        if key.find('acc') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')
    plt.show()



    

    plot_log('log.csv'),show=None)
