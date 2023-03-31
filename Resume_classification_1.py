#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import spacy
import pickle
import random
import doc
from spacy import displacy
#!pip install spacy
import warnings
warnings.filterwarnings("ignore")
import glob
import docx
import glob
import warnings
warnings.filterwarnings("ignore")
from spacy import displacy
import docx
from spacy import schemas
from spacy import Dict
from spacy.lang.en.stop_words import  STOP_WORDS
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import textract
#import antiword
from PyPDF2 import PdfFileReader
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from spacy.matcher import Matcher

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_sm")
import nltk
from spacy.matcher import Matcher


# ## Import Data
# ## For Sql developer resumes

# In[2]:


path='C:\EXCELR CLASSROOM\DATA_SCIENCE\PROJECT_207\Resumes\SQL Developer Lightning insight'
all_files=glob.glob(path + "/*.docx")
all_files


# In[3]:


def readtxt(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


# In[4]:



li=[]
for filename in all_files:
    dummy_1=readtxt(filename)
    li.append(dummy_1)
    


# In[5]:


import pandas as pd


# In[6]:


dataframe=pd.DataFrame()


# In[7]:


for files in all_files:
    print(files)
    dataframe["cv"]=li


# In[8]:


dataframe


# In[9]:


label_list=[]
for i in range (len(all_files)):
    label="SQLDeveloper"
    label_list.append(label)
    


# In[10]:


dataframe["label"]=label_list


# In[11]:


dataframe


# ## For workday resumes

# In[12]:


path1='C:\EXCELR CLASSROOM\DATA_SCIENCE\PROJECT_207\Resumes\workday resumes'
all_files1=glob.glob(path1 + "/*.docx")
all_files1


# In[13]:


li1=[]
for filename1 in all_files1:
    dummy_11=readtxt(filename1)
    li1.append(dummy_11)


# In[14]:


dataframe1=pd.DataFrame()


# In[15]:


for files in all_files1:
    print(files)
    dataframe1["cv"]=li1


# In[16]:


label_list1=[]
for i in range (len(all_files1)):
    label1="workdayResumes"
    label_list1.append(label1)
    


# In[17]:


dataframe1["label"]=label_list1


# In[18]:


dataframe1


# ## For peoplesoft resumes

# In[19]:


path2='C:\EXCELR CLASSROOM\DATA_SCIENCE\PROJECT_207\Resumes\Peoplesoft resumes'
all_files2=glob.glob(path2 + "/*.docx")
all_files2


# In[20]:


li2=[]
for filename2 in all_files2:
    dummy_12=readtxt(filename2)
    li2.append(dummy_12)


# In[21]:


dataframe2=pd.DataFrame()


# In[22]:


for files in all_files2:
    print(files)
    dataframe2["cv"]=li2


# In[23]:


label_list2=[]
for i in range (len(all_files2)):
    label2="Peoplesoft"
    label_list2.append(label2)
    


# In[24]:


dataframe2["label"]=label_list2


# In[25]:


dataframe2


# ## For React resumes

# In[26]:


path3='C:\EXCELR CLASSROOM\DATA_SCIENCE\PROJECT_207\Resumes'
all_files3=glob.glob(path3 + "/*.docx")
all_files3


# In[27]:


li3=[]
for filename3 in all_files3:
    dummy_13=readtxt(filename3)
    li3.append(dummy_13)
   


# In[28]:


dataframe3=pd.DataFrame()


# In[29]:


for files in all_files3:
    print(files)
    dataframe3["cv"]=li3


# In[30]:


label_list3=[]
for i in range (len(all_files3)):
    label3="ReactDeveloper"
    label_list3.append(label3)
    


# In[31]:


dataframe3["label"]=label_list3


# In[32]:


dataframe3


# ## Final dataframe

# In[33]:


final=pd.concat([dataframe,dataframe1,dataframe2,dataframe3],axis=0)
final=final.reset_index()
final=final.drop(columns='index',axis=0)
final


# ## Different Categories

# In[34]:


final.label.value_counts()


# In[35]:


print ("Displaying the distinct categories of resume -")
print (final.label.unique())


# In[36]:


print ("Displaying the distinct categories of resume and the number of records belonging to each category -")
print (final.label
       .value_counts())


# ## Bar Plot

# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(15,7))
plt.title("The distinct categories of resumes")
plt.xticks(rotation=90)
sns.countplot(y="label", data=final,color=None)
plt.show()


# In[38]:


import numpy as np


# ## Pie Plot

# In[39]:


from matplotlib.gridspec import GridSpec
targetCounts = final.label.value_counts()
targetLabels  = final.label.unique()
# Make square figures and axes
plt.figure(1, figsize=(15,15))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, 6)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()


# In[40]:


import plotly.express as px


# In[41]:


px.pie(data_frame=final,names="label",values=None,hover_name=None,
    hover_data=None,hole=0.05,)


# In[42]:


import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# ## Exploratory Data Analysis
# ### Text PreProcessing

# In[43]:


stopwords=list(STOP_WORDS)
stopwords


# In[44]:


punct = string.punctuation
print(punct)


# ## Cleaning Resume Text

# ## Segmentation and Lemmatisation and Normalisation
# ### Create a Function

# In[45]:


clean = []
lz = WordNetLemmatizer()
for i in range(final.shape[0]):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        final["cv"].iloc[i],
    )
    review = re.sub(r"[0-9]+", " ", review) # Remove Numbers
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [ lz.lemmatize(word) for word in review if word not in STOP_WORDS]
    review = " ".join(review)
    clean.append(review)


# In[46]:


final["Clean_Resume"] = clean


# In[47]:


final


# In[48]:


final["Clean_Resume"][0]


# ## NER (Name Entity Recognition) Using Inbuilt Function of Spacy

# In[49]:


nlp = spacy.load("en_core_web_sm")


# In[50]:


text=nlp(final["Clean_Resume"][1])


# In[51]:


displacy.render(text, style = "ent")


# In[52]:


for ent in text.ents:
  print(f'{ent.label_.upper():{20}} - {ent.text}')


# ## Creating WordCloud

# In[53]:


import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r"[0-9]+", " ", resumeText)
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


# In[54]:


import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = final.Clean_Resume
cleanedSentences = ""
for i in range(len(final.Clean_Resume)):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Most Frequent Words

# In[55]:


mostcommon = wordfreqdist.most_common(50)
print(mostcommon)


# In[56]:


from collections import Counter


# In[57]:


import seaborn as sns


# In[58]:


def wordBarGraphFunction(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.style.use('fivethirtyeight')
    plt.barh(range(20), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])],color=["g","b","r","m"])
    plt.yticks([x + 0.5 for x in range(20)], reversed(popular_words_nonstop[0:20]))
    plt.title(title)
    plt.show()


# In[59]:


plt.figure(figsize=(10,8))
wordBarGraphFunction(final,"Clean_Resume","Most frequent Words ")


# In[60]:


def wordBarGraphFunction_1(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.style.use('fivethirtyeight')
    sns.barplot(x=np.arange(20),y= [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])
    plt.xticks([x + 0.5 for x in range(20)], reversed(popular_words_nonstop[0:20]),rotation=90)
    plt.title(title)
    plt.show()


# In[61]:


plt.figure(figsize=(12,6))
wordBarGraphFunction_1(final,"Clean_Resume","Most frequent Words ")


# ## Final DataFrame

# In[62]:


resume_data=pd.DataFrame()


# In[63]:


resume_data["Resume"]=final["Clean_Resume"]
resume_data["category"]=final["label"]


# In[64]:


resume_data


# ## Labeling

# In[65]:


from sklearn.preprocessing import LabelEncoder
le_encoder=LabelEncoder()


# In[66]:


resume_data["Encoded_Skill"]=le_encoder.fit_transform(resume_data["category"])


# In[67]:


resume_data


# In[68]:


# saving the dataframe
#resume_data.to_csv('resume.csv')


# In[69]:


resume_data.isna().sum()


# In[70]:


resume_data.describe()


# In[71]:


resume_data.info()


# ## Vectorization

# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[73]:


requiredText = resume_data["Resume"].values
requiredTarget = resume_data["Encoded_Skill"].values


# ## TF-IDF

# In[74]:


word_vectorizer = TfidfVectorizer(smooth_idf=True,analyzer='word',ngram_range=(1,3),
    sublinear_tf=True,
    stop_words='english',
    max_features=5000)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,stratify=requiredTarget,shuffle = True,random_state=20, test_size=0.3)
print(X_train.shape)
print(X_test.shape)


# In[75]:


X_train.shape,y_train.shape


# In[76]:


print("X_train:\n",X_train)
print("---------------------------------------------")
print("X_test:\n",X_test)


# ## Model Building

# In[77]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly as py
from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
import plotly.graph_objs as go


# ## Support Vector Machine

# In[78]:


svm = OneVsRestClassifier(SVC(C=1.0,kernel='linear',degree=3,gamma='scale',class_weight ='balanced'))
svm.fit(X_train, y_train)
svm_prediction = svm.predict(X_test)
svm_score = svm.score(X_test, y_test)
print("SVM Classification Train Accuracy: {}%".format(round(svm.score(X_train,y_train)*100,2)))
print("SVM Classification Test Accuracy: {}%".format(round(svm.score(X_test,y_test)*100,2)))
svm_cm = confusion_matrix(y_test, svm_prediction)

print("Classification Report:\n")

print(classification_report(y_test, svm_prediction))


# ### Naive Bayes Classifier

# In[79]:


clf =MultinomialNB(alpha=1, fit_prior=False, class_prior=None).fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of MultinomialNB Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of MultinomialNB Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
nb_score = clf.score(X_test, y_test)
nb_cm = confusion_matrix(y_test, prediction)


# ## Decision Tree Classifier

# In[80]:


dt = DecisionTreeClassifier(criterion='entropy',class_weight = "balanced",splitter='best',max_depth=None)
dt.fit(X_train, y_train)
dt_prediction = dt.predict(X_test)
dt_score = dt.score(X_test, y_test)
print("Decision Tree Classification Train Accuracy: {}%".format(round(dt.score(X_train,y_train)*100,2)))
print("Decision Tree Classification Test Accuracy: {}%".format(round(dt.score(X_test,y_test)*100,2)))
dt_cm = confusion_matrix(y_test, dt_prediction)

print("Classification Report:\n")

print(classification_report(y_test, dt_prediction))


# ## KNeighbors Classifier

# In[81]:


knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3, weights='uniform',p=2,metric='minkowski',algorithm='brute'))
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)
knn_score = knn.score(X_test, y_test)
print("KNN Classification Train Accuracy: {}%".format(round(knn.score(X_train,y_train)*100,2)))
print("KNN Classification Test Accuracy: {}%".format(round(knn.score(X_test,y_test)*100,2)))
knn_cm = confusion_matrix(y_test, knn_prediction)

print('Classification Report:\n')

print(classification_report(y_test, knn_prediction))
vis = (classification_report(y_test, knn_prediction))


# In[82]:


#Find Best K Value

score_list = []
for each in range(1,30):
    knn2 =OneVsRestClassifier(KNeighborsClassifier(n_neighbors=each, weights='uniform',p=2,metric='minkowski',algorithm='brute'))
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))
plt.plot(range(1,30), score_list)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()


# ## Random Forest Classifier

# In[83]:


rf =RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=3,max_features='auto',random_state=None,
 class_weight="balanced")
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
rf_score = rf.score(X_test, y_test)
print("Random Forest Classification Train Accuracy: {}%".format(round(rf.score(X_train,y_train)*100,2)))
print("Random Forest Classification Test Accuracy: {}%".format(round(rf.score(X_test,y_test)*100,2)))
rf_cm = confusion_matrix(y_test, rf_prediction)
print("Classification Report:\n")
print(classification_report(y_test, rf_prediction))


# ### Hyper Parameter Tweaking:
# ### Grid_Search CV

# In[84]:


from sklearn.model_selection import GridSearchCV

grid_search_ = GridSearchCV(estimator =rf,param_grid = {'criterion':['entropy','gini'],
                                                                'max_depth':[1,2,3,4,5,6,7,8,9,10]},
                              cv=5)
grid_search_.fit(X_train,y_train)
print(grid_search_.best_params_)
print(grid_search_.best_score_)


# In[85]:


train_accuracy_1=[]
test_accuracy_1=[]
for depth in range(1,10):
    model_2=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=depth,max_features='auto',random_state=None,
 class_weight="balanced")
    model_2.fit(X_train,y_train)
    train_accuracy_1.append(model_2.score(X_train,y_train))
    test_accuracy_1.append(model_2.score(X_test,y_test))


# In[86]:


frame_1=pd.DataFrame({'max_depth':range(1,10),'train_acc':train_accuracy_1,'test_acc':test_accuracy_1})
plt.figure(figsize=(10,6))
plt.plot(frame_1["max_depth"],frame_1["train_acc"],marker='o')
plt.plot(frame_1["max_depth"],frame_1["test_acc"])
plt.xlabel("depth of tree")
plt.ylabel("performance")


# ## AdaBoost Classifier

# In[87]:


from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
ab_clf = AdaBoostClassifier()
ab_clf.fit(X_train, y_train)
ab_score= ab_clf.score(X_test, y_test)
ab_prediction = ab_clf.predict(X_test)
print("AdaBoost Classifier Train Accuracy: {}%".format(round(ab_clf.score(X_train,y_train)*100,2)))
print("AdaBoost Classifier Test Accuracy: {}%".format(round(ab_clf.score(X_test,y_test)*100,2)))      
ab_cm = confusion_matrix(y_test, ab_prediction)
print("Classification Report:\n")
print(classification_report(y_test, ab_prediction))


# ## Gradient Boosting Classifier

# In[88]:


gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
gb_score=gb_clf.score(X_test, y_test)
gb_prediction = gb_clf.predict(X_test)
print("Gradient Boosting Classifier Train Accuracy: {}%".format(round(gb_clf.score(X_train,y_train)*100,2)))
print("Gradient Boosting Classifier Test Accuracy: {}%".format(round(gb_clf.score(X_test,y_test)*100,2)))      
gb_cm = confusion_matrix(y_test, gb_prediction)
print("Classification Report:\n")
print(classification_report(y_test, gb_prediction))


# In[89]:


get_ipython().system('pip install lightgbm')


# ## Xtreme Gradient Boosting Classifier

# In[90]:


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_score=xgb_clf.score(X_test, y_test)
xgb_prediction = xgb_clf.predict(X_test)
print("Xtreme Gradient Boosting Classifier Train Accuracy: {}%".format(round(xgb_clf.score(X_train,y_train)*100,2)))
print("Xtreme Gradient Boosting Classifier Test Accuracy: {}%".format(round(xgb_clf.score(X_test,y_test)*100,2)))      
xgb_cm = confusion_matrix(y_test, xgb_prediction)
print("Classification Report:\n")
print(classification_report(y_test, xgb_prediction))


# ## Light Gradient Boosting Classifier

# In[91]:


lgb_clf = LGBMClassifier()
lgb_clf.fit(X_train, y_train)
lgb_score=lgb_clf.score(X_test, y_test)
lgb_prediction = lgb_clf.predict(X_test)
print("Light Gradient Boosting Classifier Train Accuracy: {}%".format(round(lgb_clf.score(X_train,y_train)*100,2)))
print("Light Gradient Boosting Classifier Test Accuracy: {}%".format(round(lgb_clf.score(X_test,y_test)*100,2)))      
lgb_cm = confusion_matrix(y_test, lgb_prediction)
print("Classification Report:\n")
print(classification_report(y_test, lgb_prediction))


# In[92]:


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix


# In[93]:


TN = [knn_cm[0,0], svm_cm[0,0], nb_cm[0,0], dt_cm[0,0], rf_cm[0,0], ab_cm[0,0], gb_cm[0,0], xgb_cm[0,0], lgb_cm[0,0]]
FP = [knn_cm[0,1], svm_cm[0,1], nb_cm[0,1], dt_cm[0,1], rf_cm[0,1], ab_cm[0,1], gb_cm[0,1], xgb_cm[0,1], lgb_cm[0,1]]
FN = [knn_cm[1,0], svm_cm[1,0], nb_cm[1,0], dt_cm[1,0], rf_cm[1,0], ab_cm[1,0], gb_cm[1,0], xgb_cm[1,0], lgb_cm[1,0]]
TP = [knn_cm[1,1], svm_cm[1,1], nb_cm[1,1], dt_cm[1,1], rf_cm[1,1], ab_cm[1,1], gb_cm[1,1], xgb_cm[1,1], lgb_cm[1,1]]
Accuracy = [knn_score, svm_score, nb_score, dt_score, rf_score, ab_score, gb_score, xgb_score, lgb_score]
Classification = ["KNN Classification", "SVM Classification", "Naive Bayes Classification", 
                  "Decision Tree Classification", "Random Forest Classification","AdaBoost Classifier","GradientBoosting Classifier",
                   "Xtreme Gradient Boosting Classifier", "Light Gradient Boosting Classifier"]
list_matrix = [Classification, TN, FP, FN, TP, Accuracy]
list_headers = ["Model", "TN", "FP", "FN", "TP", "Accuracy"]
zipped = list(zip(list_headers, list_matrix))
data_dict = dict(zipped)
df_1=pd.DataFrame(data_dict)


# In[94]:


df_1


# In[95]:


#Accuracy
plt.figure(figsize=(10,5))
ax= sns.barplot(x=df_1.Model, y=df_1.Accuracy, palette =sns.color_palette("husl", 9) )
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.style.use('seaborn-dark-palette')
plt.xlabel('Classification Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores of Classification Models')
for i in ax.patches:
    ax.text(i.get_x()+.19, i.get_height()-0.3,             str(round((i.get_height()), 4)), fontsize=15, color='b')
plt.show()
#sns.cubehelix_palette(len(df_1.Model)


# import pickle
# with open('resume_classification.pickle','wb') as f:
#     pickle.dump(knn,f)

# import pickle
# with open('model_tf.pickle','wb') as f:
#     pickle.dump(word_vectorizer,f)

# In[ ]:




