
## startup2vec

or yet another xxx2vec

check it out: 
## http://www.3top.com/startup2vec

This is a company search engine. It is built using a latent space of companies descriptions. Each company description is taken from [Crunchbase](http://www.crunchbase.com), and word embeddings from the Stanford [GloVe vectors](http://nlp.stanford.edu/projects/glove/) are aggregated to form company vectors. 

A pretty straightforward implementation, here are the steps:
 1. Load the company descriptions in a pandas frame
 2. parse the description and
 3. Get and combine their word vectors

Using 
 * [Gensim](https://radimrehurek.com/gensim/) for reading the GloVe vectors
 * [spaCy](http://spacy.io) to parse the company descriptions
 * [Pandas](pandas.pydata.org/) to load and filter company data

Now here is the code and all the details (code first, comments after). Discussion on the approach at the end.


```python
from __future__ import division
from spacy.en import English, LOCAL_DATA_DIR
from spacy.parts_of_speech import ADJ, ADV, VERB, NOUN, NAMES
import numpy as np
from gensim import matutils
from gensim.models.word2vec import Word2Vec, Vocab
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd
data_dir = os.environ.get('SPACY_DATA', LOCAL_DATA_DIR)
nlp = English(data_dir=data_dir)
```

classic pydata stuff import section...


```python
companies = pd.read_csv("notebooks/data/crunchbase/odm.csv.tar.gz", encoding='utf-8')
```


```python
startups = companies.loc[companies["primary_role"] == "company"]\
                    .loc[:,["organizations.csv", "name", "crunchbase_url",
                            "homepage_url", "profile_image_url", "short_description"]]
```

Because startups sounds much better than companies. Here I keep other columns as they will be reused on the webapp. I am not interested in other kind of companies provided by crunchbase (investment funds). 


```python
descriptions = startups[["name", "short_description"]].astype(unicode)
descriptions.drop_duplicates(subset="name", keep=False, inplace=True)
```

A little cleanup


```python
refmodel = Word2Vec.load_word2vec_format("notebooks/data/glove.840B.300d--modified.txt")
```

Loading the big GloVe model. Should be around 4GB in memory.

Now the main method parsing and aggregating vectors for each company description.

DISCLAIMER: this can probably be seriously optimized


```python
model = Word2Vec(size=300)
empty_vectors = 0
array_list = []
w_count = 0
# because I love progress bars
bar = pyprind.ProgPercent(len(descriptions), monitor=True, 
                          title="Building word2vec model based on parsed startups descriptions")
for desc in descriptions.itertuples():
    # compute the aggregated vector for the description
    name = desc[1]
    doc = nlp(desc[2]) # where the spaCy magic happens
    words = []
    vectors = []
    for t in doc:
        # keep only adjectives, verbs and nouns that are not too common
        if t.pos in (ADJ, VERB, NOUN) and t.text.lower() not in ENGLISH_STOP_WORDS:
            words.append(t.text.lower())
    for w in words:
        # get the word vector
        if w in refmodel:
            vectors.append(refmodel[w])
    # take the average vector
    vector = unitvec(np.array(vectors).mean(axis=0))
    # add the aggregated vector to the new model      
    if vector.shape:
        name = re.sub(" ", "_", name)
        model.vocab[name] = Vocab(index=w_count, count=w_count+1)
        model.index2word.append(name)
        array_list.append(vector)
        w_count += 1
        assert len(model.vocab) == len(array_list)
    else:
        empty_vectors += 1
    bar.update()
# just found out that using vstack in a loop is very good to create slow running programs
# so
model.syn0 = np.vstack(array_list)
print "{} empty vectors for a total of {}".format(empty_vectors, len(descriptions))
```

That's it. On the application side either load the model in a numpy array or in gensim if you want to use its convenient helper methods.

Because we have a complex infrastructure with django running on multiple threads and we don't want to overload the memory, I use the same [Word Embeddings as a Service](http://nbviewer.ipython.org/github/lechatpito/PyDataNYC2015/blob/master/Word%20embeddings%20as%20a%20service%20-%20PyData%20NYC%202015%20%20.ipynb) approach we are using for querying the GloVe model, through the [Word2vec API](https://github.com/3Top/word2vec-api/). 

I have experimented a few approaches before finding this is the one that works best. 

At first I wanted to try Doc2Vec as it preserves words order. I followed the [Gensim tutorial](https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb) using companies descriptions only (versus using an external model for word vectors). Results where not convincing but there is still a possibility of building document vectors with the pretrained word vectors. To investigate more. 

Then I made my way to the current solution starting with a simple word average on the description, without parsing. Lots of crap. Adding parsing made it better, but the real kick was filtering using stop words. I guess very common verbs like "are" or "is" did introduce noise. I did not yet try to do only the stop word filtering part without parsing. This could be interesting as spaCy, while being blazing fast does introduce a significant slowdown.

The next big improvement to the search engine would be to introduce companies categories to filter results. They are however not included in the Crunchbase Open Data Map and Crunchbase offered me a modest \$5000 a month to use their full API... Crunchbase if you are reading this, understand how cool it would be to have a category filter. 

Check it out: http://www.3top.com/startup2vec

