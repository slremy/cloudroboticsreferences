from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np

import tensorflow as tf

from nltk.tokenize import RegexpTokenizer

data_index = 0

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = map(uint8, array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
    
    return pl_colorscale

def plot_n_save(data, positions, thecolors, name):
    my_cmap = cm.get_cmap('jet')
    cscale = matplotlib_to_plotly(my_cmap, 255)
    index = arange(0,data.shape[0],1);
    traces = [];
    for i in unique(thecolors):
        trace = go.Scatter3d(
                             x=positions[thecolors == i,0],
                             y=positions[thecolors == i,1],
                             z=positions[thecolors == i,2],
                             mode='markers',
                             name=i,
                             text=data['title'][thecolors == i],
                             #text=["<a href=%s>%s</a>"%(i,j) for j, i in zip(data['Document Title'][thecolors == i],data[u'PDF Link'][thecolors == i])],
                             marker=dict(
                                         #color=thecolors[thecolors == i],
                                         colorscale=cscale,            # choose a colorscale
                                         size=8,
                                         opacity=1
                                         )
                             )
        traces.append(trace);
    plot(traces,filename='%s.html'%(name))

def plot2d_n_save(data, positions, thecolors, name):
    from numpy import zeros
    my_cmap = cm.get_cmap('jet')
    cscale = matplotlib_to_plotly(my_cmap, 255)
    index = arange(0,data.shape[0],1);
    traces = [];
    for i in unique(thecolors):
        trace = go.Scatter3d(
                             x=positions[thecolors == i,0],
                             z=positions[thecolors == i,1],
                             y=zeros(positions[thecolors == i].shape[0]),
                             mode='markers',
                             name=i,
                             text=data['title'][thecolors == i],
                             #text=["<a href=%s>%s</a>"%(i,j) for j, i in zip(data['Document Title'][thecolors == i],data[u'PDF Link'][thecolors == i])],
                             marker=dict(
                                         #color=thecolors[thecolors == i],
                                         colorscale=cscale,            # choose a colorscale
                                         size=8,
                                         opacity=1
                                         )
                             )
        traces.append(trace);
    plot(traces,filename='%s.html'%(name))

wordtokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b")
sentencetokenizer = RegexpTokenizer(r'[.!?]', gaps = True)

'''
def minimain:
    try:
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt

      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 500


      #data1, count1, dictionary1, reverse_dictionary1 = build_dataset(data['abstract'][0].split(), vocabulary_size)
      #val0 = pd.DataFrame([low_dim_embs[dictionary[i]] for i in "the robots are coming".split()])
      #val1 = pd.DataFrame([low_dim_embs[dictionary[i]] for i in "the robots are not coming".split()])
      #val2 = pd.DataFrame([low_dim_embs[dictionary[i]] for i in "the previous robot is good".split()])
      #val3 = pd.DataFrame([low_dim_embs[dictionary[i]] for i in "good is the robot previous".split()])
      
      viscap="Automatically  describing  the  content  of  an  image  is  a fundamental problem in artificial intelligence that connects computer vision and natural language processing.  In this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to  generate  natural  sentences  describing  an  image.   The model is trained to maximize the likelihood of the target description sentence given the training image.   Experiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions.  Our model is often quite accurate, which we verify both  qualitatively  and  quantitatively.   For  instance,  while the  current  state-of-the-art  BLEU-1  score  (the  higher  the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU-1 score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art."
      vals=[]
      for j in xrange(4):
        vals.append(pd.DataFrame([low_dim_embs[dictionary[i]] for i in mydata['abstract'][j].split()]))
      vals.append(pd.DataFrame([low_dim_embs[dictionary[i]] for i in viscap.split() if i in dictionary]))
      for dat in [val0, val1, val2, val3]: plt.plot(dat[0])
      for dat in vals: plt.plot(dat[0],'.')
      plt.legend(["the robots are coming","the robots are not coming","the previous robot is good","good is the robot previous"])

      low_dim_embs = tsne.fit_transform(final_embeddings)
      labels = [reverse_dictionary[i] for i in xrange(plot_only)]
      plot_with_labels(low_dim_embs, labels)

    except ImportError:
      print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
'''

def build_dataset(words, vocabulary_size):
      count = [['UNK', -1]]
      count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
      print(len(count))
      dictionary = dict()
      for word, _ in count:
        dictionary[word] = len(dictionary)
      data = list()
      unk_count = 0
      for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count += 1
        data.append(index)
      count[0][1] = unk_count
      #print(count)
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      return data, count, dictionary, reverse_dictionary

    # Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
      global data_index
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1  # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
          while target in targets_to_avoid:
            target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      # Backtrack a little bit to avoid skipping words in the end of a batch
      data_index = (data_index + len(data) - span) % len(data)
      return batch, labels


def transform(alldata):
    from sys import exc_info
    global data_index

    mysentences = sentencetokenizer.tokenize("".join(alldata))
    mywords = wordtokenizer.tokenize("".join(alldata))
    for i in range(len(mywords)):
        mywords[i] = mywords[i].lower()
    vocabulary_size = len(unique(mywords)) #4712 #4652
    
    data, count, dictionary, reverse_dictionary = build_dataset(mywords, vocabulary_size)
    data_index = 0

    # Step 4: Build and train a skip-gram model.

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 50     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      init = tf.global_variables_initializer()

    # Step 5: Begin training.
    num_steps = 50001#100001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print("Initialized")

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch( data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print("Average loss at step ", step, ": ", average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              try:
                close_word = reverse_dictionary[nearest[k]]
              except:
                print(exc_info()[0], k, i, nearest, log_str)
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()
    return final_embeddings, dictionary, reverse_dictionary, count


def transform_abstracts(alldata, final_embeddings, dictionary, reverse_dictionary):
    from sys import exc_info
    # Step 6: Visualize the embeddings.
    pca=PCA(n_components=2, random_state=1);
    plotlen = 500;#len(reverse_dictionary)
    low_dim_embs = pca.fit_transform(final_embeddings[:plotlen])
    labels = [reverse_dictionary[i].encode('utf-8') for i in xrange(plotlen)]
    plot_with_labels(low_dim_embs, labels)

    embabs = []
    for abstract in alldata:
        mysentences = sentencetokenizer.tokenize(abstract)
        embsentences =  []
        for l in range(len(mysentences)):
                sentence = mysentences[l]
                try:
                    embsentences.append(array([low_dim_embs[dictionary[i.lower()]] for i in wordtokenizer.tokenize(sentence)]).flatten())
                except:
                    print(exc_info(), sentence)
        embabs.append(array(embsentences))

    embabstracts = array(embabs)
    senences = sentencetokenizer.tokenize("".join(mydata['abstract']))
    words = wordtokenizer.tokenize("".join(mydata['abstract']))

    return embabstracts;


'''

[low_dim_embs[dictionary[i.lower()]] for sentence in sentences for i in sentence.split()]
[[i.lower()] for sentence in sentences for i in sentence.split()]
[i for sentence in sentences for i in sentence.split()]

for j in xrange(4):
    vals.append(
                pd.DataFrame([low_dim_embs[dictionary[i]] for i in j.split() for j in sentences])
                )

data, count, dictionary, reverse_dictionary = build_dataset(mywords, vocabulary_size)

pdata, pcount, pdictionary, preverse_dictionary = build_dataset(array(embabstracts).flatten(), vocabulary_size)

data_index = 0

'''
