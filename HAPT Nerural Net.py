import numpy as np
import pandas as pd

# display pandas results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.float_format', lambda y: '%.3f' % y)

#Reading the file with feature names
with open('G:/Machine Learning/Hapt project/HAPT Data Set/features.txt') as handle:
    features = handle.readlines()
    features = list(map(lambda x: x.strip(), features))
    
#Reading the file with class names
with open('G:/Machine Learning/Hapt project/HAPT Data Set/activity_labels.txt') as file:
    activity_labels = file.readlines()
    activity_labels = list(map(lambda y: y.strip(), activity_labels))
    
activity_data = pd.DataFrame(activity_labels)
activity_data = pd.DataFrame(activity_data[0].str.split(' ').tolist(),
                           columns = ['activity_id', 'activity_label'])


X_train = pd.read_csv('G:/Machine Learning/Hapt project/HAPT Data Set/Train/X_train.txt', delimiter = ' ', names = features)
y_train = pd.read_csv('G:/Machine Learning/Hapt project/HAPT Data Set/Train/y_train.txt', delimiter = " ", names = ['activity_id'])

X_test = pd.read_csv('/Machine Learning/Hapt project/HAPT Data Set/Test/X_test.txt', delimiter = " ", names = features)
y_test = pd.read_csv('/Machine Learning/Hapt project/HAPT Data Set/Test/y_test.txt', delimiter = " ", names = ['activity_id'])

# Joining the dataset
X_merge = [X_train,X_test]
y_merge = [y_train,y_test]
X = pd.concat(X_merge)
y = pd.concat(y_merge)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Stats Models doesn't consider the bias term so we add a column of ones at the beginning.
# New statsmodels lib only has the syntax of statsmodel.api
import statsmodels.api as sm

X = np.append(arr = np.ones((10929,1)).astype(int),values = X,axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y , x).fit()
        
        maxVar = max(regressor_OLS.pvalues) #.astype(float)
        
        if maxVar > sl:
            for j in range(0, numVars - i):
                
                if (regressor_OLS.pvalues[j]== maxVar):#.astype(float) 
                    
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Standard Limit
SL = 0.075
columns = list(range(0,561))
X_opt = X[:,0:]
X_final = backwardElimination(X_opt, SL) 

# Test for Adj R2
regressor_OLS = sm.OLS(y, X_final).fit() # sl = 7.5% r-adj = 98.4
regressor_OLS.summary()
regressor_OLS.rsquared

from sklearn.model_selection import train_test_split
X_val_train,X_val_test,y_val_train,y_val_test = train_test_split(X_final,y,test_size = .1)

from sklearn.model_selection import train_test_split
X_train_final,X_test_final,y_train_final,y_test_final = train_test_split(X_val_train,y_val_train,test_size = .1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit(X_train_final)
X = sc.transform(X_train_final)
X = sc.transform(X_val_test)
X = sc.transform(X_test_final)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()

y_train_final = one.fit_transform(y_train_final)
y_train_final = y_train_final.toarray()

y_test_final = one.fit_transform(y_test_final)
y_test_final = y_test_final.toarray()

y_val_test = one.fit_transform(y_val_test)
y_val_test = y_val_test.toarray()


# Building the neural net.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

neural = Sequential()

# first hidden layer
neural.add(Dense(units = 38,input_shape = [266,],activation='relu',kernel_initializer='glorot_uniform'))
neural.add(Dropout(rate = .3))

# Second hidden layer
#neural.add(Dense(units =32,activation='relu',kernel_initializer='glorot_uniform'))               
#neural.add(Dropout(rate = .4))

# Output layer
neural.add(Dense(units = 12,activation='softmax',kernel_initializer='glorot_uniform'))

# Compiling the ANN (Applying the SGD )
#sgd = optimizers.SGD(lr=[0.1], decay=1e-7, momentum=0.9, nesterov=True)
#sgd = optimizers.SGD(lr=[0.01], decay=1e-7, momentum=0.9, nesterov=True)
#sgd = optimizers.SGD(lr=[0.001], decay=1e-7, momentum=0.9, nesterov=True)

#sgd = optimizers.SGD(lr=[0.1], decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=[0.01], decay=1e-7, momentum=0.8, nesterov=True)
#sgd = optimizers.SGD(lr=[0.001], decay=1e-6, momentum=0.9, nesterov=True)

neural.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
              
# Training the model.
neural.fit(X_train_final,y_train_final,batch_size = 20, epochs = 25) # acc = 95.48% dropout 40%

# model performance..
val_score=(neural.evaluate(X_val_test, y_val_test, batch_size=10, verbose=1, sample_weight=None, steps=None)) # 95.05%

test_score=(neural.evaluate(X_test_final, y_test_final, batch_size=10, verbose=1, sample_weight=None, steps=None)) # 94.4%

y_pred = neural.predict(X_test_final)
y_pred = (y_pred > .5)

# Making the multiclass confusion matrix
from sklearn.metrics import multilabel_confusion_matrix, classification_report
cm = multilabel_confusion_matrix(y_test_final,y_pred)

#Final resulting metrics
print(classification_report(y_test_final,y_pred, target_names = activity_labels))






