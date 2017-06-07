from sys import exc_info, argv
import pandas as pd
from pybtex.database.input import bibtex as bibtex_in
import textwrap

bibparser = bibtex_in.Parser()

d=[]
for bibfile in argv[1:]:
    #bib=bibparser.parse_file(bibfile)
    with open(bibfile) as f:
        raw = f.read()
        f.close()
    bib=bibparser.parse_string(raw)#raw.encode(encoding='UTF-8',errors='strict')
    for bib_id in bib.entries:
        b = bib.entries[bib_id].fields
        #if hasattr(b,"booktitle"): print b["booktitle"], bib_id
        if b["abstract"] == "" or 'keywords' not in b.keys(): continue
        #print b
        d.append([" <br> ".join(textwrap.wrap(b["title"],60)),
                  b["abstract"],
                  b["year"],
                  #b["booktitle"] if "booktitle" in b else b["journal"],
                  b["keywords"],
                  bib.entries[bib_id].type])

mydata = pd.DataFrame(d, columns = ["title", "abstract", "year", "keywords","document identifier"])

from plotfcns import *


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, normalize
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

#vectorizer = CountVectorizer(min_df=1)
# or
vectorizer = TfidfVectorizer(max_df=0.5, max_features=100000,
                             min_df=2, stop_words='english',
                             use_idf=True)
X = vectorizer.fit_transform(mydata['abstract'])

pca=PCA(n_components=3, random_state=1);
X = pca.fit_transform(X.toarray())
'''
svd = TruncatedSVD(3)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
'''
#for i in unique(mydata['year']): print(i, mydata[mydata['year']==i].shape[0])



badtitle = ["In the News", "Program guide", "Table of", "Title page", "Front co", "Copyright not", "Content list", "Proceedings", "Contents","Cover art"]
plot_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X, mydata['document identifier'], "type")
#plot2d_n_save(mydata[~mydata['title'].str.contains('|'.join(badtitle),case=False)], X, mydata['document identifier'], "type")
