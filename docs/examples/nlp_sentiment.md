# NLP x Sentiment Analysis

In this tutorial, you will learn how to leverage Modelkit as a support for a common NLP task: Sentiment Analysis.

You will see how Modelkit can be useful as you progress along the following steps:

1. Implementing the Tokenizer leveraging Spacy
2. Implementing the Vectorizer leveraging Scikit-Learn
3. Building a simple Classifier leveraging Keras to predict whether a review is negative or positive

Last but not least, you will get to see how a breaze it is to speed up your go-to-production process with Modelkit. 

------
## Tokenizer

In this section, we will cover the basics of Modelkit's API, and use Spacy as tokenizer for our NLP pipeline.

### Initialization
Once you have set up a fresh python environment, let's install modelkit, spacy and grab the small english model.

```
pip install modelkit spacy
python -m spacy download en_core_web_sm
```

### Basics

To define a Modelkit Model, you need to:

- create class inheriting from `modelkit.Model` 
- implement a `_predict` method

To begin with, let's create a minimal tokenizer in the most concise way:

```python
import modelkit


class Tokenizer(modelkit.Model):
    def _predict(self, text):
        return text.split()
```

That's it ! As you can see, it is _very_ minimal, and probably not sufficient for a lot of usages.

It is just the necessary to define a Modelkit Model.

You are now able to instantiate and call it using the following methods:
```python
tokenizer = Tokenizer()

# for single predictions
tokenizer("I am a Data Scientist from Amiens, France")
tokenizer.predict("I am a Data Scientist from Amiens, France")

# for batch predictions
tokenizer.predict_batch(["I am a Data Scientist from Amiens, France"])

# for generators
tokenizer.predict_gen(("I am a Data Scientist from Amiens, France",))

```

### Leveraging spacy

For now, the tokenizer pipeline is too simple to show-off.

Let's make use of Spacy and complexify things a little, so that to have a prod-like tokenizer. (fortunately, we will not be digging into weird customer rules)

This will also help demonstrate additional Modelkit features.

```python
import modelkit
import spacy

class Tokenizer(modelkit.Model):
    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict(self, text):
        text = " ".join(text.replace("<br", "").replace("/>", "").split())
        return [
            t.lower_
            for t in self.nlp(text)
            if t.is_ascii and len(t) > 1 and not (t.is_punct or t.is_stop or t.is_digit)
        ]
```

Spacy pipelines come with several different cool features.

Since we will only be using the tokenizer, let's not forget to disable the ones we will not be needing.

As you can see, the `_load` method was implemented: it is where all the assets, artefacts, models etc. are loaded once the model is instantiated.

This way, your model will benefit from further features that will be covered in the next sections such as *lazy_loading* and *dependency management*.

This is why we store the spacy pipeline in the `_load` method, not in the `_predict` nor the `__init__` methods.

```python
tokenizer = Tokenizer()
tokenizer.predict("Spacy is a great lib for NLP 😀")
# ['spacy', 'great', 'lib', 'nlp']

```

### Batch computing

So far, we have only implemented the `_predict` method so that to tokenize one sentence at a time.

While this is relevant for many usages, when calls are rather simple and straightforward (such as the `text.split()` tokenizer),
it's a whole other thing when you deal with more complex functions leveraging heavy weapons (Spacy, Tensorflow, PyTorch etc.) or distant calls (TF Serving, database accesses etc.) 

To do so, Modelkit allows you to define a `_predict_batch` method to _kill N birds with one stone_.

The following section shows how to properly use Spacy's `pipe` method to tokenize items by batch.

```python
import modelkit
import spacy


class Tokenizer(modelkit.Model):
    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict_batch(self, texts):
        texts = [
            " ".join(text.replace("<br", "").replace("/>", "").split())
            for text in texts
        ]
        return [
            [
                t.lower_
                for t in text
                if t.is_ascii
                and len(t) > 1
                and not (t.is_punct or t.is_stop or t.is_digit)
            ]
            for text in self.nlp.pipe(texts, batch_size=len(texts))
        ]
```

This way, we reduce by two the time needed to tokenize batches of data, compared to a simple `_predict` call.

Benchmark example using ipython:
```
%timeit [Tokenizer().predict("Spacy is a great lib for NLP") for _ in range(100)]
# 11.1 ms ± 203 µs per loop on a 2020 Macbook Pro.

%timeit Tokenizer().predict_batch(["Spacy is a great lib for NLP] * 100, batch_size=64)
# 5.5 ms ± 105 µs per loop on a 2020 Macbook Pro.
```

### Testing

So far, more or less complex rules have been designed.

Modelkit allows you to add tests right next to your class definition to serve as documentation, in addition to ensuring everything behaves as intended.

```python
import modelkit
import spacy


class Tokenizer(modelkit.Model):
    TEST_CASES = {
        "cases": [
            {"item": "", "result": []},
            {"item": "NLP 101", "result": ["nlp"]},
            {
                "item": "I'm loving the Spacy 101 course !!!ù*`^@😀",
                "result": ["loving", "spacy", "course"],
            },
            {
                "item": "<br/>prepare things for IMDB<br/>",
                "result": ["prepare", "things", "imdb"],
            },
            {
                "item": "<br/>a b c data<br/>      e scientist",
                "result": ["data", "scientist", "failing", "test"],
            },  # fails as intended
        ]
    }

    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict_batch(self, texts):
        texts = [
            " ".join(text.replace("<br", "").replace("/>", "").split())
            for text in texts
        ]
        return [
            [
                t.lower_
                for t in text
                if t.is_ascii
                and len(t) > 1
                and not (t.is_punct or t.is_stop or t.is_digit)
            ]
            for text in self.nlp.pipe(texts, batch_size=len(texts))
        ]
```

These tests can be executed in two ways.

Manually, in an interative programming tool such as ipython, jupyter etc. using the `test` method:
```python
Tokenizer().test()
# TEST 1: SUCCESS
# TEST 2: SUCCESS
# TEST 3: SUCCESS
# TEST 4: SUCCESS
# TEST 5: FAILED test failed on item
# item = '<br/>a b c data<br/>      e scientist'                                               
# expected = list instance                                                                     
# result = list instance                                                                       
```

Or with pytest via [the Modelkit autotesting fixture](../library/testing.html#modellibrary-fixture-and-autotesting)

Woops, seems like we need to fix the last test!

### Typing

When it comes to production, it is good practice to type models' inputs and outputs to ensure consistency and validation between calls, dependencies and services.

It also allows you to better/faster understand how to use a given model, in addition to benefit from static type checkers such as mypy.

Modelkit manages typing in the following way: `Model[input_type, output_type]`

It also support the power of Pydantic, as we will see in the next section.

Let's type our previous Tokenizer and conclude this first part:

```python
from typing import List

import modelkit
import spacy


class Tokenizer(modelkit.Model[str, List[str]]):
    TEST_CASES = {
        "cases": [
            {"item": "", "result": []},
            {"item": "NLP 101", "result": ["nlp"]},
            {
                "item": "I'm loving the Spacy 101 course !!!ù*`^@😀",
                "result": ["loving", "spacy", "course"],
            },
            {
                "item": "<br/>prepare things for IMDB<br/>",
                "result": ["prepare", "things", "imdb"],
            },
            {
                "item": "<br/>a b c data<br/>      e scientist",
                "result": ["data", "scientist", "failing", "test"],
            },  # fails as intended
        ]
    }

    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict_batch(self, texts):
        texts = [
            " ".join(text.replace("<br", "").replace("/>", "").split())
            for text in texts
        ]
        return [
            [
                t.lower_
                for t in text
                if t.is_ascii
                and len(t) > 1
                and not (t.is_punct or t.is_stop or t.is_digit)
            ]
            for text in self.nlp.pipe(texts, batch_size=len(texts))
        ]
```

This way, the following will raise a Modelkit `ItemValidationException`:

```python
Tokenizer().predict(["list", "of", "tokens", "will", "raise", "an", "exception"])
```

That's it! To sum up this first Modelkit overview, you have learned:

- How to create a basic Model class by inheriting from `modelkit.Model` and implementing a `_predict` method
- How to correctly load artefacts/assets by overloading the `_load` method
- How to leverage batch computing to speed up execution by implementing a `_predict_batch` method
- How to add tests to ensure everything works as intended using `TEST_CASES` right in your model definition
- How to add typing to ensure consistency and validation: `modelkit.Model[input_type, output_type]`

### Final tokenizer

```python
from typing import List

import modelkit
import spacy


class Tokenizer(modelkit.Model[str, List[str]]):
    CONFIGURATIONS = {"imdb_tokenizer": {}}
    TEST_CASES = {
        "cases": [
            {"item": "", "result": []},
            {"item": "NLP 101", "result": ["nlp"]},
            {
                "item": "I'm loving the Spacy 101 course !!!ù*`^@😀",
                "result": ["loving", "spacy", "course"],
            },
            {
                "item": "<br/>prepare things for IMDB<br/>",
                "result": ["prepare", "things", "imdb"],
            },
            {
                "item": "<br/>a b c data<br/>      e scientist",
                "result": ["data", "scientist"],
            },
        ]
    }

    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict_batch(self, texts):
        texts = [
            " ".join(text.replace("<br", "").replace("/>", "").split())
            for text in texts
        ]
        return [
            [
                t.lower_
                for t in text
                if t.is_ascii
                and len(t) > 1
                and not (t.is_punct or t.is_stop or t.is_digit)
            ]
            for text in self.nlp.pipe(texts, batch_size=len(texts))
        ]

```
As you may have seen, there is a `CONFIGURATIONS` map in the class definition, we will cover it in the next section.

------
## Vectorizer

In this section, we will fit and implement a custom text vectorizer based on the sentiment analysis IMDB dataset.

It will be the opportunity to go through Modelkit's Assets Management basics, to learn how to fetch artefacts, in addition to getting even more familiar with its API.

### Initialization

To begin with, let's download the IMDB reviews dataset:

```
# download the remote archive
curl https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz --output imdb.tar.gz
# untar it
tar -xvf imdb.tar.gz
# remove the unsupervised directory we will not be using
rm -rf aclImdb/train/unsup
```

This dataset is organized in train/test directories, containing pos (positive) / neg (negative) subfolders corresponding to the target classes to predict.

The different files only consist in longer or shorter text reviews left by IMDB users, one file corresponding to one review.

### Fitting

First thing first, let's define a helper function to read through the training set.

We will only be using generators to avoid loading up the entire dataset in memory, and take advantage of the `predict_gen` method of Modelkit models.

```python 
import glob
from typing import Generator

def read_dataset(path: str) -> Generator[str, None, None]:
    for review in glob.glob(os.path.join(path, "*.txt")):
        with open(review, 'r', encoding='utf-8') as f:
            yield f.read()
```

Before fitting our Scikit-Learn `TfidfVectorizer`, we need to tokenize our dataset first.

While this vectorizer includes a tokenizer, we will be using the one implemented in the first section.

```python
import itertools
import os

training_set = itertools.chain(
    read_dataset(os.path.join("aclImdb", "train", "pos")),
    read_dataset(os.path.join("aclImdb", "train", "neg")),
)
tokenized_set = Tokenizer().predict_gen(training_set)
```

By using the `predict_gen` method, we are able to read huge texts corpora without having Out-Of-Memory issues.

Each review will be tokenized one by one. As you may remember, you can also speed up execution by setting a `batch_size` greater than one to process these reviews batch by batch.

```python
# here, an example with a 64 batch size
tokenized_set = Tokenizer().predict_gen(training_set, batch_size=64)
```

We are all set! Let's fit our `TfidfVectorizer` using the tokenized_set, and disabling the embedded tokenizer.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x, lowercase=False, max_df=0.95, min_df=0.01
).fit(tokenized_set)
```

Then, we will just keep the just-fitted vocabulary and dump it to disk, before writing our on vectorizer.

Of course, one could stick with the TfidfVectorizer, however the idea here is not to rely on it for production for the following reasons:

- its goal here is just to help build the vocabulary using neat features such as min/max document frequencies
- it would require a heavy dependency: scikit-learn, just for vectorization
- pickling scikit-learn models come with some tricky issues relative to dependency, version and security management that we will not be digging here

It is also common practice to separate the research / training phase, from the production phase.

As such, we will be implementing our own vectorizer for production using Modelkit, based on the work of scikit-learn's `TfidfVectorizer`.

All in all, we just need to dump a list of strings to disk.

```python
# we only keep strings from the vocabulary
# we will be using our own str -> int mapping
vocabulary = next(zip(*sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])))
with open("vocabulary.txt", "w", encoding="utf-8") as f:
    for row in vocabulary:
        f.write(row + "\n")
```

### Asset retrieval

In this subsection, you will learn how to leverage the basics of Modelkit's Assets Management system.

Keep in mind that most of its powerful features (assets pushing, updating, versioning, retrieval) will not be adressed in this tutorial, *but are the way to go when things get serious.*

First, let's implement our Vectorizer as a Modelkit Model, the same way we did for the Tokenizer:

- inheriting from Model, implementing the `_predict` method
- typing ins and outs
- testing its behavior
- dealing with fixed sizes and out-of-vocabulary tokens

```python
import numpy as np
import modelkit

from typing import List


class Vectorizer(modelkit.Model[List[str], List[int]]):
    TEST_CASES = {
        "cases": [
            {"item": [], "result": []},
            {"item": [], "keyword_args": {"length": 10}, "result": [0] * 10},
            {"item": ["movie"], "result": [888]},
            {"item": ["unknown_token"], "result": []},
            {
                "item": ["unknown_token"],
                "keyword_args": {"drop_oov": False},
                "result": [1],
            },
            {"item": ["movie", "unknown_token", "scenes"], "result": [888, 1156]},
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"drop_oov": False},
                "result": [888, 1, 1156],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10},
                "result": [888, 1156, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10, "drop_oov": False},
                "result": [888, 1, 1156, 0, 0, 0, 0, 0, 0, 0],
            },
        ]
    }


    def _predict(self, tokens, length=None, drop_oov=True):
        vectorized = (
            np.array(self._vectorizer(tokens), dtype=np.int)
            if tokens
            else np.array([], dtype=int)
        )
        if drop_oov and len(vectorized):
            vectorized = np.delete(vectorized, vectorized == 1)
        if not length:
            return vectorized.tolist()
        result = np.zeros(length)
        vectorized = vectorized[:length]
        result[: len(vectorized)] = vectorized
        return result.tolist()
``` 

As you can see here, there are some missing pieces in the puzzle.

However, you probably noticed one new piece in TEST_CASES: `keyword_args`.

This allows you to pass extra parameters to your `_predict`, in this case: `length`, `drop_oov`

Let's figure out what the other pieces are.

To begin with, you need to be introduced to the `CONFIGURATIONS` map:

```python
class Vectorizer(modelkit.Model[List[str], List[int]]):
    CONFIGURATIONS = {
        "imdb_vectorizer": {
            "asset": "vocabulary.txt"
        }
    }
    ...
```

The `CONFIGURATIONS` map is the cornerstone of the Assets Management system.
It is the best way to:

- name models
- use local/remote/versioned assets
- make models depend on other models
- use different assets, with the same implementation
- get models, by their name using Modelkit's `ModelLibrary`

In the previous code example, you can see that we named our vectorizer `imdb_vectorizer`, it relies on a local asset whose direct path is `vocabulary.txt`.

What about the `_load` method ? Let's implement it: 

```python
import numpy as np


class Vectorizer(modelkit.Model[List[str], List[int]]):
    CONFIGURATIONS = {"imdb_vectorizer": {"asset": "vocabulary.txt"}}
    ...
    def _load(self):
        self.vocabulary = {}
        with open(self.asset_path, "r", encoding="utf-8") as f:
            for i, k in enumerate(f):
                self.vocabulary[k.strip()] = i + 2
        self._vectorizer = np.vectorize(lambda x: self.vocabulary.get(x, 1))

```

Right before the `_load` method is called, the local/remote/versioned asset is retrieved and cached on the caller's disk, and its local path is set in the `self.asset_path` attribute.

In this case, it is rather straighforward: `self.asset_path` directly points to `vocabulary.txt`.

Hence, we just need to iterate through it, populate our vocabulary dictionary and map those tokens to their number of line + 2:

- 0 for padding
- 1 for the unknown token (out-of-vocabulary words)
- 2 for everything else

Finally, the `np.vectorize` will help map a tokens list to an indices list.

### Final vectorizer

Here is the entire `Vectorizer` puzzle:

```python 
import modelkit
import numpy as np

from typing import List


class Vectorizer(modelkit.Model[List[str], List[int]]):
    CONFIGURATIONS = {"imdb_vectorizer": {"asset": "vocabulary.txt"}}
    TEST_CASES = {
        "cases": [
            {"item": [], "result": []},
            {"item": [], "keyword_args": {"length": 10}, "result": [0] * 10},
            {"item": ["movie"], "result": [888]},
            {"item": ["unknown_token"], "result": []},
            {
                "item": ["unknown_token"],
                "keyword_args": {"drop_oov": False},
                "result": [1],
            },
            {"item": ["movie", "unknown_token", "scenes"], "result": [888, 1156]},
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"drop_oov": False},
                "result": [888, 1, 1156],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10},
                "result": [888, 1156, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10, "drop_oov": False},
                "result": [888, 1, 1156, 0, 0, 0, 0, 0, 0, 0],
            },
        ]
    }

    def _load(self):
        self.vocabulary = {}
        with open(self.asset_path, "r", encoding="utf-8") as f:
            for i, k in enumerate(f):
                self.vocabulary[k.strip()] = i + 2
        self._vectorizer = np.vectorize(lambda x: self.vocabulary.get(x, 1))

    def _predict(self, tokens, length=None, drop_oov=True):
        vectorized = (
            np.array(self._vectorizer(tokens), dtype=np.int)
            if tokens
            else np.array([], dtype=int)
        )
        if drop_oov and len(vectorized):
            vectorized = np.delete(vectorized, vectorized == 1)
        if not length:
            return vectorized.tolist()
        result = np.zeros(length)
        vectorized = vectorized[:length]
        result[: len(vectorized)] = vectorized
        return result.tolist()
```

------

## Classifier

In this section, we will be implementing a simple sentiment classifier leveraging Keras, plugging the different components we have been developping so far.

### Data processing

We will reuse the `read_dataset` function, in addition to a few other helper functions 
which will continue taking advantage of generators to avoid loading up the entire dataset into memory:

- `alternate(f, g)`: to yield items alternatively between the f and g generators 
- `alternate_labels()`: to yield 1 and 0, alternatively, in sync with the previous `alternate` function


```python
import glob 
import os

from typing import Generator, Any

def read_dataset(path: str) -> Generator[str, None, None]:
    for review in glob.glob(os.path.join(path, "*.txt")):
        with open(review, 'r', encoding='utf-8') as f:
            yield f.read()

def alternate(
    f: Generator[Any, Any, Any], g: Generator[Any, Any, Any]
) -> Generator[Any, None, None]:
    while True:
        try:
            yield next(f)
            yield next(g)
        except StopIteration:
            break

def alternate_labels() -> Generator[int, None, None]:
    while True:
        yield 1
        yield 0

```

Let's plug these pipes together so that to efficiently read and process the IMDB reviews dataset for our next Keras classifier, leveraging the Tokenizer and Vectorizer we implemented in the first two sections:

```python
def process(path, tokenizer, vectorizer, length, batch_size):
    # read the positive sentiment reviews dataset
    positive_dataset = read_dataset(os.path.join(path, "pos"))
    # read the negative sentiment reviews dataset
    negative_dataset = read_dataset(os.path.join(path, "neg"))
    # alternate between positives and negatives examples
    dataset = alternate(positive_dataset, negative_dataset)
    # generate labels in sync with the previous generator: 1 for positive examples, 0 for negative ones
    labels = alternate_labels()

    # tokenize the reviews using our Tokenizer Model
    tokenized_dataset = tokenizer.predict_gen(dataset, batch_size=batch_size)
    # vectorize the reviews using our Vectorizer Model
    vectorized_dataset = vectorizer.predict_gen(
        tokenized_dataset, length=length, batch_size=batch_size
    )
    # yield (review, label) tuples for Keras
    yield from zip(vectorized_dataset, labels)
```

Let's try it out on the first examples:
```python

i = 0
for review, label in process(
    os.path.join("aclImdb", "train"),
    Tokenizer(),
    Vectorizer(),
    length=64,
    batch_size=64,
):
    print(review, label)
    i += 1
    if i >= 10:
        break
```
### Model library

So far, we have only be using instantiating the `Tokenizer` and `Vectorizer` classes as standard objects.

Modelkit provides another, simpler and better way to instantiate those: `modelkit.ModelLibrary`.

Its different features make the usage of `ModelLibrary` the best way to load models:

- you can fetch models by their `CONFIGURATIONS` name using the `get` method
- two calls to `get` will not create two models, only one
- it supports prediction caching: [Prediction Caching](../library/model_library.html#prediction-caching)
- it allows you to lazy load your models: [Lazy Loading](../library/model_library.html#lazy-loading)
- you can parametrize your models through it: [Settings](../library/model_library.html#modellibrary)
- it encourages you to keep your models organized:  [Organizing Models](../library/organizing.html)

We will not deal with package organization in this tutorial, so let's see how we can take advantage of the `ModelLibrary` with our previous work.


```python
import modelkit

model_library = modelkit.ModelLibrary(models=[Vectorizer, Tokenizer])
tokenizer = model_library.get("imdb_tokenizer")
vectorizer = model_library.get("imdb_vectorizer")

```

Another tutorial will cover how to package these models as a python package, and how to benefit even more from Modelkit's `ModelLibrary` powerful features.

### Keras training

We now need to create a `TF Dataset` from our data processing generators:
 
```python
import os
import tensorflow as tf

BATCH_SIZE = 64
LENGTH = 64

training_set = (
    tf.data.Dataset.from_generator(
        lambda: process(
            os.path.join("aclImdb", "train"),
            tokenizer,
            vectorizer,
            length=LENGTH,
            batch_size=BATCH_SIZE,
        ),
        output_types=(tf.int16, tf.int16),
    )
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(1)
)
validation_set = (
    tf.data.Dataset.from_generator(
        lambda: process(
            os.path.join("aclImdb", "test"),
            tokenizer,
            vectorizer,
            length=LENGTH,
            batch_size=BATCH_SIZE,
        ),
        output_types=(tf.int16, tf.int16),
    )
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(1)
)


```

This is it for the data processing part. 

Let's train a basic Keras classifier to predict whether an IMDB review is positive or negative, and save it to disk.

```python
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=len(vectorizer.vocabulary) + 2, output_dim=64, input_length=LENGTH
        ),
        tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.binary_accuracy],
)
model.build()
model.fit(
    training_set,
    validation_data=validation_set,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=10,
)
model.save(
    "imdb_model.h5", include_optimizer=False, save_format="h5", save_traces=False
)
```

### Final classifier 
Voila ! As we already did for the Vectorizer, we will embed the just-saved `imdb_model.h5` in a basic Modelkit `Model` which we will further upgrade in the next section.

```python
import modelkit
import tensorflow as tf

from typing import List


class Classifier(modelkit.Model[List[int], float]):
    CONFIGURATIONS = {"imdb_classifier": {"asset": "imdb_model.h5"}}

    def _load(self):
        self.model = tf.keras.models.load_model(self.asset_path)

    def _predict_batch(self, vectorized_reviews):
        return self.model.predict(vectorized_reviews)
```

The same way we did for the Vectorizer, the `MovieReviewToSentiment` model has a `movie_review_to_sentiment` configuration which points to the `imdb_model.h5`.

We also benefit from Keras' `predict` ability to batch predictions in our `_predict_batch` method.


### End-to-end example

To sum up, this is one way we can chain together the Tokenizer, Vectorizer and Classifier:


```python
import modelkit

library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
tokenizer = library.get("imdb_tokenizer")
vectorizer = library.get("imdb_vectorizer")
classifier = library.get("imdb_classifier")

review = "I freaking love this movie, the main character is so cool !"

tokenized_review = tokenizer(review)  # or tokenizer.predict
vectorized_review = vectorizer(tokenized_review)  # or vectorizer.predict
prediction = classifier(vectorized_review)  # or classifier.predict
```

## Wiring things up

You have now seen most dev features, implementing them from the Tokenizer to the Classifier.

While it works from end-to-end, the API is still not really user-friendly and asks to wire things up as we did in the last section.

Moreover, the classifier still seems kind of basic, and does not implement many features that make Modelkit relevant.

Let's see how we can enhance it, and take the opportuniy to review all Modelkit's features we have seen so far, and discover new features too.

### Dependency management

Modelkit's Asset Management system allows you to add other models as dependencies.

For our use-case, it might be relevant for the classifier to take raw string reviews as input, and output a class label ("good" or "bad").

This could be done using the `model_dependencies` key in the `CONFIGURATIONS` map, and by adding the `Tokenizer` and the `Vectorizer` as dependencies.

Modelkit takes care of loading up the dependencies right before your model's instantiation.
As such, loaded dependencies are discoverable from `self.model_dependencies` anytime after the `_load` method is called.

```python
import numpy as np
import modelkit
import tensorflow as tf


class Classifier(modelkit.Model[str, str]):
    CONFIGURATIONS = {
        "imdb_classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {"imdb_tokenizer", "imdb_vectorizer"}
        },
    }
    def _load(self):
        self.model = tf.keras.models.load_model(self.asset_path)
        self.tokenizer = self.model_dependencies["imdb_tokenizer"]
        self.vectorizer = self.model_dependencies["imdb_vectorizer"]
        self.prediction_mapper = np.vectorize(lambda x: "good" if x >= 0.5 else "bad")

    def _predict_batch(self, reviews):
        # this looks like the previous end-to-end example from the previous section
        cleaned_reviews = self.tokenizer.predict_batch(reviews)
        vectorized_reviews = self.vectorizer.predict_batch(cleaned_reviews, length=64)
        predictions_scores = self.model.predict(vectorized_reviews)
        predictions_classes = self.prediction_mapper(predictions_scores).reshape(-1)
        return predictions_classes


model_library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
classifier = model_library.get("imdb_classifier")
classifier.predict("This movie is freaking awesome, I love the main character")
# good
classifier.predict("this movie sucks, it's the worst I have ever seen")
# bad
```
This definitely looks easier to use, and the classifier now seems more relevant.

### Tests, again

We might want to add the score, in addition to some tests as well:

```python
import numpy as np
import modelkit
import tensorflow as tf

from typing import Dict, Union

class Classifier(modelkit.Model[str, Dict[str, Union[str, float]]]):
    CONFIGURATIONS = {
        "imdb_classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {"imdb_tokenizer", "imdb_vectorizer"},
        },
    }
    TEST_CASES = {
        "cases": [
            {
                "item": "i love this film, it's the best i've ever seen",
                "result": {"score": 0.8441019058227539, "label": "good"},
            },
            {
                "item": "this movie sucks, it's the worst I have ever seen",
                "result": {"score": 0.1625385582447052, "label": "bad"},
            },
        ]
    }

    def _load(self):
        self.model = tf.keras.models.load_model(self.asset_path)
        self.tokenizer = self.model_dependencies["imdb_tokenizer"]
        self.vectorizer = self.model_dependencies["imdb_vectorizer"]
        self.prediction_mapper = np.vectorize(lambda x: "good" if x >= 0.5 else "bad")

    def _predict_batch(self, reviews):
        cleaned_reviews = self.tokenizer.predict_batch(reviews)
        vectorized_reviews = self.vectorizer.predict_batch(cleaned_reviews, length=64)
        predictions_scores = self.model.predict(vectorized_reviews)
        predictions_classes = self.prediction_mapper(predictions_scores).reshape(-1)
        predictions = [
            {"score": score, "label": label}
            for score, label in zip(predictions_scores, predictions_classes)
        ]
        return predictions


model_library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
classifier = model_library.get("imdb_classifier")
classifier.test()
```

### Pydantic

We now output a score and its corresponding label in a `Dict`.

For production usages, Modelkit can leverage the power of Pydantic to validate both input and output items, instead of the standard python way.

It is even more relevant, readable and understandable as your number of input / output features grow (such as a `rating` field in the next example)

The only usage difference: inputs and outputs are now python objects and need to be managed as such.


```python

from typing import Optional

import modelkit
import numpy as np
import pydantic
import tensorflow as tf


class MovieReviewItem(pydantic.BaseModel):
    text: str
    rating: Optional[float] = None  # could be useful in the future ? but not mandatory

class MovieSentimentItem(pydantic.BaseModel):
    label: str
    score: float

class Classifier(modelkit.Model[MovieReviewItem, MovieSentimentItem]):
    CONFIGURATIONS = {
        "imdb_classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {"imdb_tokenizer", "imdb_vectorizer"},
        },
    }
    TEST_CASES = {
        "cases": [
            {
                "item": {"text": "i love this film, it's the best I've ever seen"},
                "result": {"score": 0.8441019058227539, "label": "good"},
            },
            {
                "item": {"text": "this movie sucks, it's the worst I have ever seen"},
                "result": {"score": 0.1625385582447052, "label": "bad"},
            },
        ]
    }

    def _load(self):
        self.model = tf.keras.models.load_model(self.asset_path)
        self.tokenizer = self.model_dependencies["imdb_tokenizer"]
        self.vectorizer = self.model_dependencies["imdb_vectorizer"]
        self.prediction_mapper = np.vectorize(lambda x: "good" if x >= 0.5 else "bad")

    def _predict_batch(self, reviews):
        texts = (review.text for review in reviews)
        cleaned_reviews = self.tokenizer.predict_batch(texts)
        vectorized_reviews = self.vectorizer.predict_batch(cleaned_reviews, length=64)
        predictions_scores = self.model.predict(vectorized_reviews)
        predictions_classes = self.prediction_mapper(predictions_scores).reshape(-1)
        predictions = [
            {"score": score, "label": label}
            for score, label in zip(predictions_scores, predictions_classes)
        ]
        return predictions

model_library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
classifier = model_library.get("imdb_classifier")
classifier.test()
prediction = classifier.predict({"text": "I love the main character"})
print(prediction)
# MovieSentimentItem(label='good', score=0.6801363825798035)
print(prediction.label) 
# good
print(prediction.score) 
# 0.6801363825798035
```

You can also rename your model dependencies from within the `CONFIGURATIONS` map, 
so that to make some tricky dependencies names more understandable, of if you do not know their exact name in advance:

```python
import modelkit

class Classifier(modelkit.Model[MovieReviewItem, MovieSentimentItem]):
    CONFIGURATIONS = {
        "imdb_classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {
                "tokenizer": "imdb_tokenizer",
                "vectorizer": "imdb_vectorizer",
            },  # renaming dependencies here
        },
    }
    def _load(self):
        self.tokenizer = self.model_dependencies[
            "tokenizer"
        ]
        self.vectorizer = self.model_dependencies[
            "vectorizer"
        ]
    ...

```

### Multiple configurations

To conclude this tutorial, let's briefly see how to define multiple configurations, and why.

What if this model went to production, most clients are happy with it, but some are not. 
You start from scratch: redefining your tokenizer, then your vectorizer, 
then re-ran your classifier's training loop with a different architecture and more data, and saved it preciously.
Since you have always been the original guy in the room, all the new models now have a "_SOTA" suffix.

You managed to keep the same inputs, outputs, pipeline architecture and made your process reproductible.

"That should do it !", you are yelling across the open space.

However, you probably do not want to surprise your clients with a new model without informing them before.

Some might want to stick with the old one, while the unhappy ones would want to change ASAP.

Modelkit has got your back, and allows you to define multiple configurations while keeping the exact same code.

Here is how you would process:

```python
import modelkit

class Classifier(modelkit.Model[MovieReviewItem, MovieSentimentItem]):
    CONFIGURATIONS = {
        "classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {
                "tokenizer": "imdb_tokenizer",
                "vectorizer": "imdb_vectorizer",
            },
            "test_cases": [
                {
                    "item": {"text": "i love this film, it's the best i've ever seen"},
                    "result": {"score": 0.8441019058227539, "label": "good"},
                },
                {
                    "item": {
                        "text": "this movie sucks, it's the worst I have ever seen"
                    },
                    "result": {"score": 0.1625385582447052, "label": "bad"},
                },
            ],
        },
        "classifier_SOTA": {
            "asset": "imdb_model_SOTA.h5",
            "model_dependencies": {
                "tokenizer": "imdb_tokenizer_SOTA",
                "vectorizer": "imdb_vectorizer_SOTA",
            },
            "test_cases": [
                {
                    "item": {"text": "i love this film, it's the best i've ever seen"},
                    "result": {"score": 1.0, "label": "good"},
                },
                {
                    "item": {
                        "text": "this movie sucks, it's the worst I have ever seen"
                    },
                    "result": {"score": 0.0, "label": "bad"},
                },
            ],
        },
    }
    ...

model_library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
classifier_deprecated = model_library.get("imdb_classifier")
classifier_SOTA = model_library.get("imdb_classifier_SOTA")
```

As you can see, the `CONFIGURATIONS` map and the dependency renaming helped make this task easier than we may have thought in the first instance.

Also, the `tests_cases` are now part of each individual configuration so that to test each one independently.

## TL;DR

Here is the entire tutorial implementation, covering most Modelkit's features to get you started.

```python
from typing import List, Optional

import modelkit
import numpy as np
import pydantic
import spacy
import tensorflow as tf


class Tokenizer(modelkit.Model[str, List[str]]):
    CONFIGURATIONS = {"imdb_tokenizer": {}}
    TEST_CASES = {
        "cases": [
            {"item": "", "result": []},
            {"item": "NLP 101", "result": ["nlp"]},
            {
                "item": "I'm loving the Spacy 101 course !!!ù*`^@😀",
                "result": ["loving", "spacy", "course"],
            },
            {
                "item": "<br/>prepare things for IMDB<br/>",
                "result": ["prepare", "things", "imdb"],
            },
            {
                "item": "<br/>a b c data<br/>      e scientist",
                "result": ["data", "scientist"],
            },
        ]
    }

    def _load(self):
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "attribute_ruler",
            ],
        )

    def _predict_batch(self, texts):
        texts = [
            " ".join(text.replace("<br", "").replace("/>", "").split())
            for text in texts
        ]
        return [
            [
                t.lower_
                for t in text
                if t.is_ascii
                and len(t) > 1
                and not (t.is_punct or t.is_stop or t.is_digit)
            ]
            for text in self.nlp.pipe(texts, batch_size=len(texts))
        ]


class Vectorizer(modelkit.Model[List[str], List[int]]):
    CONFIGURATIONS = {"imdb_vectorizer": {"asset": "vocabulary.txt"}}
    TEST_CASES = {
        "cases": [
            {"item": [], "result": []},
            {"item": [], "keyword_args": {"length": 10}, "result": [0] * 10},
            {"item": ["movie"], "result": [888]},
            {"item": ["unknown_token"], "result": []},
            {
                "item": ["unknown_token"],
                "keyword_args": {"drop_oov": False},
                "result": [1],
            },
            {"item": ["movie", "unknown_token", "scenes"], "result": [888, 1156]},
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"drop_oov": False},
                "result": [888, 1, 1156],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10},
                "result": [888, 1156, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            {
                "item": ["movie", "unknown_token", "scenes"],
                "keyword_args": {"length": 10, "drop_oov": False},
                "result": [888, 1, 1156, 0, 0, 0, 0, 0, 0, 0],
            },
        ]
    }

    def _load(self):
        self.vocabulary = {}
        with open(self.asset_path, "r", encoding="utf-8") as f:
            for i, k in enumerate(f):
                self.vocabulary[k.strip()] = i + 2
        self._vectorizer = np.vectorize(lambda x: self.vocabulary.get(x, 1))

    def _predict(self, tokens, length=None, drop_oov=True):
        vectorized = (
            np.array(self._vectorizer(tokens), dtype=np.int)
            if tokens
            else np.array([], dtype=int)
        )
        if drop_oov and len(vectorized):
            vectorized = np.delete(vectorized, vectorized == 1)
        if not length:
            return vectorized.tolist()
        result = np.zeros(length)
        vectorized = vectorized[:length]
        result[: len(vectorized)] = vectorized
        return result.tolist()


class MovieReviewItem(pydantic.BaseModel):
    text: str
    rating: Optional[
        float
    ] = None  # could be useful in the future ? but not mandatory


class MovieSentimentItem(pydantic.BaseModel):
    label: str
    score: float


class Classifier(modelkit.Model[MovieReviewItem, MovieSentimentItem]):
    CONFIGURATIONS = {
        "imdb_classifier": {
            "asset": "imdb_model.h5",
            "model_dependencies": {
                "tokenizer": "imdb_tokenizer",
                "vectorizer": "imdb_vectorizer",
            },
        },
    }
    TEST_CASES = {
        "cases": [
            {
                "item": {"text": "i love this film, it's the best I've ever seen"},
                "result": {"score": 0.8441019058227539, "label": "good"},
            },
            {
                "item": {"text": "this movie sucks, it's the worst I have ever seen"},
                "result": {"score": 0.1625385582447052, "label": "bad"},
            },
        ]
    }

    def _load(self):
        self.model = tf.keras.models.load_model(self.asset_path)
        self.tokenizer = self.model_dependencies["tokenizer"]
        self.vectorizer = self.model_dependencies["vectorizer"]
        self.prediction_mapper = np.vectorize(lambda x: "good" if x >= 0.5 else "bad")

    def _predict_batch(self, reviews):
        texts = (review.text for review in reviews)
        cleaned_reviews = self.tokenizer.predict_batch(texts)
        vectorized_reviews = self.vectorizer.predict_batch(cleaned_reviews, length=64)
        predictions_scores = self.model.predict(vectorized_reviews)
        predictions_classes = self.prediction_mapper(predictions_scores).reshape(-1)
        predictions = [
            {"score": score, "label": label}
            for score, label in zip(predictions_scores, predictions_classes)
        ]
        return predictions


model_library = modelkit.ModelLibrary(models=[Tokenizer, Vectorizer, Classifier])
classifier = model_library.get("imdb_classifier")
prediction = classifier.predict({"text": "I love the main character"})
print(prediction)
# good
```
