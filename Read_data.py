# read tabular data and preparing features and targets

''' read data to features -x and targets - y
x, y =
'''

#################### spliting train, valid and test sets ####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y[:, -1])  # stratified split on y group
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.18, random_state=42, stratify=y_train[:,-1])
print(X_train.shape, X_valid.shape, X_test.shape)
#################### scaling features #################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
#################### binarize Labels #################################
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_valid = lb.fit_transform(y_valid)
y_test = lb.fit_transform(y_test)

