import pandas as pd
data=pd.read_csv("C://Users//User//Downloads//Courses.csv")
data=data.dropna(how='any')
data['Description']=data['Description'].replace({"'ll":" "},regex=True)
data['CourseId']=data['CourseId'].replace({"-":" "},regex=True)

comb_frame=data.CourseId.str.cat(" "+data.CourseTitle.str.cat(" "+data.Description))
comb_frame=data.CourseId.str.cat(" "+data.CourseTitle.str.cat(" "+data.Description))
comb_frame=comb_frame.replace({"[^A-Za-z0-9]+": ""},regex=True)

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)
true_k = 8
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=15)
model.fit(X)
doc_centroids=model.cluster_centers_.argsort()[:, ::-1]
terms=vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" %i),
    for ind in doc_centroids[i,:15]:
        print(' %s ' %terms[ind]),
    print
