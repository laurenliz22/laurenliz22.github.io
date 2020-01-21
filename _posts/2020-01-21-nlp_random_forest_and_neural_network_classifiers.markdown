---
layout: post
title:      "NLP: Random Forest & Neural Network Classifiers"
date:       2020-01-21 18:01:09 +0000
permalink:  nlp_random_forest_and_neural_network_classifiers
---


After cleaning and exploring my dataset for my NLP project, I wanted to model my data using both a Random Forest Classifier as well as a Neural Network Classifier.  To prepare the data for these models I had to take a couple of different methods.  After a lot of googling, I thought it would be helpful to describe these methods in a cohesive blog!

**Feature Engineering**

To prepare my dataset for modeling, I used both Word2Vec and TF-IDF to vectorize my features.  In my case, these were tweets that I was classifying into an existing account category.  

*Word2Vec*

I started with Word2Vec as my vectorization strategy because my dataset was so large.  Word2Vec is a model that is pretrained on a very large corpus and provides embeddings that map words that are similar or close to each other.  

* Step 1: Develop my train/validate and test data sets for modeling
```
#Define X and y.  My tweets have already been tokenized here.
import numpy as np
X = np.array(clean_data.tokens)
y = np.array(clean_data.account_category)
```
```
#Create Train/Validate/Test data splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=40)
```

* Step 2: Before I feed my tokens into a Word2Vec model I will turn them into LabeledSentence objects below
```
#labelize tweets
import gensim
from tqdm import tqdm
LabeledSentence = gensim.models.doc2vec.LabeledSentence
tqdm.pandas(desc="progress-bar") #estimate time to completion
```
```
#return labelized tweets
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
```
```
#split labelized tweets by train/test/split data
X_train = labelizeTweets(X_train, 'TRAIN')
X_val = labelizeTweets(X_val, 'VALIDATE')
X_test = labelizeTweets(X_test, 'TEST')
```
I'll check out the first item of my labelized X_train data to confirm this worked
```
X_train[0]
```
Output:
<blockquote class="imgur-embed-pub" lang="en" data-id="a/ce9oMUt" data-context="false" ><a href="//imgur.com/a/ce9oMUt"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
* Step 3: Now each element is an object with two attributes: a list of tokens and a label. The next step is to build and train my Word2Vec model.
```
#Build the Word2Vec Model
from gensim.models.word2vec import Word2Vec 
#initialize model, a typical size is between 100-300
tweet_w2v = Word2Vec(size=200, window = 5, min_count=10, workers=4) 
#create vocabulary
tweet_w2v.build_vocab([x.words for x in tqdm(X_train)]) 
#train model
tweet_w2v.train([x.words for x in tqdm(X_train)], total_examples=tweet_w2v.corpus_count, epochs=2) 
```
When I check to make sure that my Word2Vec Model worked correctly, I should obtain a vector that is 200-dimensions.  An example to see a vector for the word 'happy' is `tweet_w2v['happy']` and to view similar words along with their probability to the word 'happy' is `tweet_w2v.most_similar('happy')`
* Step 4: Now let's visualize results using the bokeh library for interactive visualization and tsne for converting the vectors to 2D vectors
```
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from sklearn.manifold import TSNE
```
```
# defining the chart
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)
```
```
# getting a list of word vectors. limit to 10,000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]
```
```
# dimensionality reduction. converting the vectors to 2d vectors
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)
```
```
# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]
```
```
# plotting the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)
```
Output:
<blockquote class="imgur-embed-pub" lang="en" data-id="a/2qSvtxH" data-context="false" ><a href="//imgur.com/a/2qSvtxH"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

This is an interactive chart so as you hover over the blue dots in your notebook you'll be able to view the words!  The points/words are similar or close to each other in context. 

*TF-IDF*

In order to classify the tweets, I'll need to turn them into vectors. I know the vector representation of each word within a tweet, so I'll have to combine the vectors to get a new vector that represents the whole tweet. I've looked at different methods and it appears the best solution is to compute a weighted average with the weight as the TF-IDF score. The weight will provide the importance of the word with respect to the entire corpus.

* Step 1: Create a TF-IDF Matrix
```
# Create tf-idf matrix to produce a vocabulary size
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10) #min_df of 10 means that the word needs to show up at least 10 times in my corpus to be included in the vocabulary
matrix = vectorizer.fit_transform([x.words for x in X_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))
```

* Step 2: Produce an averaged tweet vector
```
#Build the vector producing averaged tweets
def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec
```

* Step 3: Convert X_train, X_val and X_test into list of vectors. Scale each column to have zero mean and unit standard deviation.

```
#Convert into vector and scale
from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_train))])
train_vecs_w2v = scale(train_vecs_w2v)
```
```
val_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_val))])
val_vecs_w2v = scale(val_vecs_w2v)
```
```
test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_test))])
test_vecs_w2v = scale(test_vecs_w2v)
```

Now that I have my train, validate and test sets properly vectorized, I can look at my Random Forest and Neural Network Classification Models! 

**Modeling**

*Random Forest Classification*

Using a simple Random Forest Model you can see how my model will be fit using my data sets above 

```
#Using Random Forest Classifier baseline model
from sklearn.ensemble import RandomForestClassifier
rfc =  RandomForestClassifier(n_estimators=100, verbose=True)
```
```
#Fitting baseline Random Forest Classifier
rfc.fit(train_vecs_w2v, y_train)
```

*Neural Network Classification*

I'll now feed my word2vec vectors into a neural network. I'll start with a very simple 2 layer architecture and run only 2 epochs for this example.

```
# Convert labels to categorical one-hot encoding
import pandas as pd 
ohe_y_train = pd.get_dummies(y_train)
ohe_y_val = pd.get_dummies(y_val)
ohe_y_test = pd.get_dummies(y_test)
```

```
#For a single-input model using RMSprop optimizer
import random  
from keras.models import Model, Sequential
from keras.layers import Dense
from keras import initializers, regularizers, constraints, optimizers, layers
random.seed(123)
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(200,)))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=32,
                    validation_data=(val_vecs_w2v, ohe_y_val))
```

And there you have it!  You can now continue modeling and tweaking the hyperparameters to develop the model that best fits your data!
