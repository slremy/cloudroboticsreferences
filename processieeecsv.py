from numpy import uint8, array, arange, nan, unique
from sys import exc_info, argv
import pandas as pd
import textwrap
from plotfcns import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, normalize
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

vectorizer = CountVectorizer(min_df=1)
mydata = pd.read_csv(argv[1], index_col=None, na_values=['NA'], skiprows=[0], encoding='latin-1')
mydata.columns = [i.lower() for i in mydata.columns ]
mydata['document title'] = [" <br> ".join(textwrap.wrap(i, 60)) for i in mydata['document title']]
#ieeejour =  pd.DataFrame([i for i in data.as_matrix() if 'Journals' in i[-1]], columns=data.columns)
#X = vectorizer.fit_transform(ieeejour['Abstract'])

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(mydata['abstract'])

svd = TruncatedSVD(3)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

'''
tsne=TSNE(n_components=3, random_state=1);
X =tsne.fit_transform(X.toarray())
'''
'''
pca=PCA(n_components=3, random_state=1);
X = pca.fit_transform(normalize(X.toarray()))
'''
km = MiniBatchKMeans(n_clusters=40, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000)
km.fit(X)
'''
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(40):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
np.histogram(km.labels_,40)
'''
badtitle = ["In the News", "Table of", "Title page", "Front co", "Copyright not", "Content list", "Proceedings", "Contents"]

#plot_n_save(data[~data['document title'].str.contains('|'.join(badtitle),case=False)], X, km.labels_, "kmeans")
plot_n_save(mydata[~mydata['document title'].str.contains('|'.join(badtitle),case=False)], X, mydata['year'], "years")
plot_n_save(mydata[~mydata['document title'].str.contains('|'.join(badtitle),case=False)], X, mydata['document identifier'], "type")
