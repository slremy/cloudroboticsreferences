import spacy
nlp = spacy.load('en')

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
