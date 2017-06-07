from sys import exc_info, argv
import pandas as pd
import urllib
import xml.etree.ElementTree
import textwrap

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, normalize
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

query="http://export.arxiv.org/api/query?search_query=all:cloud+AND+robotics&start=0&max_results=2000"
response = urllib.urlopen(query).read()
e = xml.etree.ElementTree.fromstring(response)

d=[[" <br> ".join(textwrap.wrap(entry.find('{http://www.w3.org/2005/Atom}title').text.replace("\n",' '),60)),entry.find('{http://www.w3.org/2005/Atom}summary').text.replace("\n",' '),entry.find('{http://www.w3.org/2005/Atom}published').text.split("-")[0],entry.findall('{http://arxiv.org/schemas/atom}primary_category')[0].attrib['term'],entry.find('{http://www.w3.org/2005/Atom}id').text.replace("\n",' ')] for entry in e.findall('{http://www.w3.org/2005/Atom}entry')]

mydata = pd.DataFrame(d, columns = ["title", "abstract", "year","document identifier","uri"])


from synnet import *
from plotfcns import *

#X = transform(mydata['abstract'])


#vectorizer = CountVectorizer(min_df=1)
# or
#vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
#                             min_df=2, stop_words='english',
#                             use_idf=True)
#X = vectorizer.fit_transform(mydata['abstract'])

#embeddings,dictionary,reverse_dictionary,count = transform(mydata['abstract'])
#pca=PCA(n_components=1, random_state=1);
#Xemb = pca.fit_transform(embeddings)

vectorizer = TfidfVectorizer(max_features=100000, stop_words=None, use_idf=True)
#vectorizer = CountVectorizer(vocabulary=dictionary)
X = vectorizer.fit_transform(mydata['abstract'])

pca=PCA(n_components=3, random_state=1);
#X1 = pca.fit_transform((X>0)*(Xemb.T*np.ones(Xemb.shape)))
X1 = pca.fit_transform(X.toarray())

#svd = TruncatedSVD(3)
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)
#X1 = lsa.fit_transform(X)



badtitle = ["In the News", "Table of", "Title page", "Front co", "Copyright not", "Content list", "Proceedings", "Contents"]
plot_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X1, mydata['document identifier'], "type")
#plot2d_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X1, mydata['document identifier'], "type")
