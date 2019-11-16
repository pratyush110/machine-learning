from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()

features=iris.data
labels=iris.target

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)

my_classifier = KNeighborsClassifier()
my_classifier.fit(features_train,labels_train)

prediction = my_classifier.predict(features_test)

print(accuracy_score(labels_test,prediction))

input_data = [[4.7, 2.5, 3.1, 1.2]]
input_prediction = my_classifier.predict(input_data)

if input_prediction[0] == 0:
    print('Setosa')

if input_prediction[0] == 1:
    print('Versicolor')

if input_prediction[0] == 2:
    print('Virginica')
