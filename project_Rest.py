import pandas as pd
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting = 3)

# cleaing text
import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# testing on a random sentence working of NLP
# creating object of Portstemmer
# ps = PorterStemmer()
# example = "This is a very Amazing dish"
# # lower case conversion
# example = example.lower()
# # splitting each word of sentence in a list
# example = example.split()
# # stemming to get root word  like amazing ->amaz
# example = [ps.stem(word) for word in example if not word in set(stopwords.words('english'))]
# example = ' '.join(example)

corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review.lower()
    review.split()
    ps = PorterStemmer()
    review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ''.join(review)
    # taking important root words from bag of words
    corpus.append(review)
# print(corpus)
# print(df)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=600)
X = cv.fit_transform(corpus).toarray()
# print(X)
y = df.iloc[:,-1].values
# print(y)
# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# fitting naive bayes algorithm to the trainig set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)
# print(y_pred)
# creating the confusion matrix
from sklearn.metrics  import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
# print(cm)
# testing accuracy
res = (cm[1][1] + cm[0][0])/200
print(res)
