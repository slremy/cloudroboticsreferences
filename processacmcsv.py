from numpy import uint8, array, arange, nan, unique
from sys import exc_info, argv
import pandas as pd
from plotfcns import *


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, normalize
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

vectorizer = CountVectorizer(min_df=1)
data = pd.read_csv(argv[1], index_col=None, encoding='latin-1')
#ieeejour =  pd.DataFrame([i for i in data.as_matrix() if 'Journals' in i[-1]], columns=data.columns)
#X = vectorizer.fit_transform(ieeejour['Abstract'])
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
#d = [i['title'] if np.isnan(i['title'] else i['booktitle'] for i in data)]
mydata = data[[isinstance(i,unicode) for i in data['title']]]
X = vectorizer.fit_transform(mydata['title'])

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

#plot_n_save(data[~data['Document Title'].str.contains('|'.join(badtitle),case=False)], X, km.labels_, "kmeans")
plot_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X, mydata['year'], "years")
#plot_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X, mydata['Document Identifier'], "type")
