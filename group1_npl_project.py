import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

#load data
path = r"D:\Centennial College Materials\THIRD SEMESTER\COMP 237 - INTRODUCTION TO AI\GROUP PROJECT" 
filename = "Youtube01-Psy.csv"
fullpath = os.path.join(path,filename)
nlp_Psy = pd.read_csv(fullpath)

#explore the data
print(nlp_Psy.head(3))
print(nlp_Psy.shape)
print(pd.DataFrame.info(nlp_Psy))

#drop columns that aren't needed
psy_col_to_drop = ['COMMENT_ID', 'AUTHOR','DATE']
nlp_Psy_a = nlp_Psy.drop(psy_col_to_drop, axis = 1)
print(pd.DataFrame.info(nlp_Psy_a))

#prepare data for transformation via count vectorizer
comments = nlp_Psy_a["CONTENT"]
label = nlp_Psy_a["CLASS"]
countVector = CountVectorizer()
train_tc = countVector.fit_transform(comments)

#shape of transformed data
print("Shape of the transformed data:", train_tc.shape)

#downscale the transformed data using tf-idf
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_tc)

#shape of final features after tf-idf transformation
print("Shape of the final features after tf-idf transformation:", train_tfidf.shape)

#shuffle the dataset using pandas.sample, setting frac=1
nlp_Psy_shuffled = nlp_Psy.sample(frac=1)


# Separate the shuffled data into training (75%) and testing (25%)
train_size = int(0.75 * len(nlp_Psy_shuffled))
train_data, test_data = nlp_Psy_shuffled[:train_size], nlp_Psy_shuffled[train_size:]

# Separate class from features
y_train = train_data['CLASS']
X_train = countVector.transform(train_data['CONTENT'])
X_train_tfidf = tfidf_transformer.transform(X_train)

# Fit the training data into a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Cross-validation on the training data using 5-fold
cv_scores = cross_val_score(nb_classifier, X_train_tfidf, y_train, cv=5)
print("Cross-validation Accuracy for 5 times", cv_scores)
print("Cross-validation Mean Accuracy:", cv_scores.mean())

# Test the model on the test data
X_test = countVector.transform(test_data['CONTENT'])
X_test_tfidf = tfidf_transformer.transform(X_test)
y_test = test_data['CLASS']

# Make predictions on the test data
predictions = nb_classifier.predict(X_test_tfidf)

# Print confusion matrix and accuracy of the model
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Test Accuracy:", accuracy_score(y_test, predictions))


#create 6 new comments and test
#I made up some comments to test model 
new_comments =["I liked his last song better", "The crowds this mad draws is amazing",
           "This concert was certainly unforgettable. My admiration for this guy PSY.",
           "Like 4 like here: https://.....", 
           "Hey everyone, I found this amazing website that gives away free gift cards! 💳💰 Just click the link in my bio to get yours now! It really works, I got a $100 gift card! 🔥🌟 #FreeGiftCards #Legit",
           "Y'all need Jesus"]
comments_vector = countVector.transform(new_comments)
comments_tfidf = tfidf_transformer.transform(comments_vector)
comments_pred = nb_classifier.predict(comments_tfidf)

#Define category map
category_map = {0: 'Non-Spam', 1: 'Spam'}

for sent, category in zip(new_comments, comments_pred):
    print('\nInput:', sent, '\nPredicted category:', category_map[category])

