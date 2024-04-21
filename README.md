# Parameter Optimisation SVM
## Ikjot Singh 
## 102116071   
## 3CS11

### Steps
- Importing Libraries
```
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
```

- Fetching the dataset from UCI repository
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
chess_king_rook_vs_king = fetch_ucirepo(id=23) 
  
# data (as pandas dataframes) 
X = chess_king_rook_vs_king.data.features 
y = chess_king_rook_vs_king.data.targets 
```

- Converting all categorical columns to numerical
```
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])
```

- Standard Scaling
```
# apply standard scaler
scaler = StandardScaler()

X = df.drop('white-depth-of-win', axis=1)
y = df['white-depth-of-win']

X = scaler.fit_transform(X)
df=pd.concat([pd.DataFrame(X),y],axis=1)
```

- Generating 10 random samples
```
samples = []
for i in range(10):
    samples.append(df.sample(n=2000))
```

- SVM Trainer
```
def train_svm(X_train,y_train,X_test,y_test, kernel ,C, gamma):
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma,max_iter=2000)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)
```

- Finding Best Parameters
```
dataframes = []
result = pd.DataFrame(columns = ['Sample Number', 'Best Kernel', 'Best C', 'Best Gamma', 'Best Accuracy'])
for num,sample in enumerate(samples, start=1):
    df = pd.DataFrame(columns = ['iteration', 'kernel', 'C', 'gamma', 'accuracy'])
    X = sample.drop(['white-depth-of-win'], axis=1)
    y = sample['white-depth-of-win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    kernelList = ['linear', 'poly', 'rbf', 'sigmoid']
    best_accuracy = 0
    for i in range(100):
        k = random.random()
        n = random.random()
        kernel = random.choice(kernelList)
        accuracy = train_svm(X_train,y_train,X_test,y_test, kernel, k, n)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_C = k
            best_gamma = n
            df = df.append({'iteration': i, 'kernel': kernel, 'C': k, 'gamma': n, 'accuracy': accuracy}, ignore_index=True)
    dataframes.append(df)
    result = result.append({'Sample Number': "sample {}".format(num), 'Best Kernel': best_kernel, 'Best C': best_C, 'Best Gamma': best_gamma, 'Best Accuracy': best_accuracy}, ignore_index=True)
```

- Results

![image](https://github.com/IkjotSingh221/Parameter-Optimisation-SVM/assets/91063550/dd48e094-92d1-4f8b-b4e0-329ad97f650e)

- Plotting the Accuracy v/s Iterations

![image](https://github.com/IkjotSingh221/Parameter-Optimisation-SVM/assets/91063550/c79d82a4-5d54-4bfc-8d8c-60f4a053c226)


### Conclusion
Accuracy increases with increase in number of iterations
