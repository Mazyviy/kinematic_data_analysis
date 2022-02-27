import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# open csv
data = pd.read_csv('Kinematics_Data.csv')
print('Исходные данные')
print(data.to_string(max_rows=10))
print()

# отбрасываем ненужную информацию
data.drop(['date', 'time', 'username'], axis=1, inplace=True)
pd.options.display.width = 0

print('Исходные данные после отбрасывания информации, которая не относится к классификации')
print(data.to_string(max_rows=10))
print()

# Корреляционная матрица
corr_matrix = data.loc[:, :].corr()
print('Корреляционная матрица')
print(corr_matrix)
print()

print('Описательные статистики')
print(data.describe())
print(f'Кол-во строк: {data.shape}')
print()

# разделяем данные
X = data.drop('activity', axis=1)
y = data['activity']

# разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
print(f'Кол-во эл. в обуч. и тест. выборках: {X_train.shape, X_test.shape}')
print()

# K-Nearest Neighbors Algorithm
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# The method of support vectors
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Linear Discriminant Analysis
lda_model = RandomForestClassifier(max_depth=2, random_state=0)
lda_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)

# время работы каждого типа классификатора
start = time.perf_counter()
knn_pred = knn_model.predict(X_test)
knn_time = time.perf_counter() - start

start = time.perf_counter()
svc_pred = svc_model.predict(X_test)
svc_time = time.perf_counter() - start

start = time.perf_counter()
dt_pred = dt_model.predict(X_test)
dt_time = time.perf_counter() - start

start = time.perf_counter()
gnb_pred = gnb_model.predict(X_test)
gnb_time = time.perf_counter() - start

start = time.perf_counter()
lda_pred = lda_model.predict(X_test)
lda_time = time.perf_counter() - start

start = time.perf_counter()
lr_pred = lr_model.predict(X_test)
lr_time = time.perf_counter() - start

print(f'Accuracy of K-Nearest Neighbor: {accuracy_score(y_test, knn_pred) * 100:.2f} %, execution time: {knn_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, knn_pred))
print('Classification report')
print(classification_report(y_test, knn_pred))

print(f'Accuracy of the support vector machine: {accuracy_score(y_test, svc_pred) * 100:.2f} %, execution time: {svc_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, svc_pred))
print('Classification report')
print(classification_report(y_test, svc_pred))

print(f'Accuracy of the decision tree classifier: {accuracy_score(y_test, dt_pred) * 100:.2f} %, execution time: {dt_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, dt_pred))
print('Classification report')
print(classification_report(y_test, dt_pred))

print(f'Accuracy of gaussian naive bayes: {accuracy_score(y_test, gnb_pred) * 100:.2f} %, execution time: {gnb_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, gnb_pred))
print('Classification report')
print(classification_report(y_test, gnb_pred))

print(f'Accuracy of linear discriminant analysis: {accuracy_score(y_test, lda_pred) * 100:.2f} %, execution time: {lda_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, lda_pred))
print('Classification report')
print(classification_report(y_test, lda_pred))

print(f'Accuracy logistic regression: {accuracy_score(y_test, lr_pred) * 100:.2f} %, execution time: {lr_time:.4f}')
print('Confusion matrix')
print(confusion_matrix(y_test, lr_pred))
print('Classification report')
print(classification_report(y_test, lr_pred))

plt.scatter(knn_time, int(accuracy_score(y_test, knn_pred) * 100), s=50, label="K-Nearest Neighbors Algorithm")
plt.scatter(svc_time, int(accuracy_score(y_test, svc_pred) * 100), s=50, label="The method of support vectors")
plt.scatter(dt_time, int(accuracy_score(y_test, dt_pred) * 100), s=50, label="Decision Tree Classifier")
plt.scatter(gnb_time, int(accuracy_score(y_test, gnb_pred) * 100), s=50, label="Gaussian Naive Bayes")
plt.scatter(lda_time, int(accuracy_score(y_test, lda_pred) * 100), s=50, label="Linear Discriminant Analysis")
plt.scatter(lr_time, int(accuracy_score(y_test, lr_pred) * 100), s=50, label="Logistic Regression")
plt.legend()
plt.ylabel('Accuracy (%)')
plt.xlabel('Prediction time (s)')
plt.show()