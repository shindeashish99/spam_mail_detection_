
#SPAM MAIL DETECTION




import pandas as pd 
import numpy as np
import nltk

messages = pd.read_csv("D:\\python\\NLP project\\spam.csv",encoding="cp1252")

messages.shape

messages = messages.iloc[:,[0,1]]


messages.rename(columns={"v1":"label","v2":"message"},inplace=True)


messages.label.value_counts()

length= messages.message.apply(len)
length

messages=pd.concat([messages,length],axis=1)


messages.columns.values[2]="Length"

messages.head


from nltk.corpus import stopwords

import string

stopwords.words("english")

string.punctuation



def text_process(mess):
    """
    1. remove the punctuation
    2. remove the stopwords
    3. return the list of clean textwords
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    
    return [ word for word in nopunc.split() if word not in stopwords.words("english")]


messages['message'].apply(text_process)

messages.message = messages.message.str.lower()
messages.head(5)




from sklearn.feature_extraction.text  import CountVectorizer

bow_transformer=CountVectorizer(analyzer=text_process).fit(messages["message"])

bow_transformer.vocabulary_

print(len(bow_transformer.vocabulary_))


messages_bow = bow_transformer.transform(messages.message)

messages_bow.shape



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(messages_bow,messages.label, test_size=0.2,random_state = 101)



from sklearn.naive_bayes import MultinomialNB
naive_bay = MultinomialNB()

spam_nb_model = naive_bay.fit(x_train,y_train)
pred=naive_bay.predict(x_test)


from pandas_ml import ConfusionMatrix
ConfusionMatrix(pred,y_test) #will not work


from sklearn.metrics import confusion_matrix
confusion_matrix(pred,y_test)


























