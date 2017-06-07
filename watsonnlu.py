import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as features

import spacy
nlp = spacy.load('en')


NLU_USERNAME = os.environ.get("NLU_USERNAME")
NLU_PASSWORD = os.environ.get("NLU_PASSWORD")

natural_language_understanding = NaturalLanguageUnderstandingV1(version='2017-02-27', username=NLU_USERNAME, password=NLU_PASSWORD)

nlu=[]
for i in xrange(mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	nlu.append(response)

for i in xrange(355,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	nlu.append(response)

for i in xrange(432,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	print i
	nlu.append(response)

for i in xrange(791,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	print i
	nlu.append(response)

for i in xrange(1069,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	print i
	nlu.append(response)

for i in xrange(1138,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	print i
	nlu.append(response)

for i in xrange(1780,mydata.shape[0]):
	response = natural_language_understanding.analyze( text = mydata['abstract'][i],features=[features.Entities(), features.Keywords(), features.Concepts()])
	print i
	nlu.append(response)


f=open("storednlu3.json","w")
json.dump(nlu,f)
f.close()
