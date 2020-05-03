# model.py
import tensorflow as tf
from tensorflow import keras as K

# path to store model weights at end of model.py
keras_model_path = 'pubmed_journal_prediction_model.h5'

# initialize key variables
embedding_dimension = 100
vocabulary_size = 24416 + 2 
# vocabulary_size = number of words in vocabulary + 2 
# extra 2 = 1 (b/c 0 doesn't mean anything) + 1 (for words that aren't in vocab)

title_input = K.Input(shape=(72,))
embedding_layer = K.layers.Embedding(vocabulary_size, embedding_dimension)
title_layer = embedding_layer(title_input)
title_layer = K.layers.Bidirectional(K.layers.GRU(64))(title_layer)
title_layer = K.layers.Dropout(0.5)(title_layer)
title_layer = K.layers.Dense(64, activation='relu')(title_layer)

abstract_input = K.Input(shape=(200,))
abstract_layer = embedding_layer(abstract_input)
abstract_layer = K.layers.Bidirectional(K.layers.GRU(64))(abstract_layer)
abstract_layer = K.layers.Dropout(0.5)(abstract_layer)
abstract_layer = K.layers.Dense(64, activation='relu')(abstract_layer)

concat = K.layers.Concatenate()([title_layer, abstract_layer])
combo = K.layers.Dense(128, activation='relu')(concat)
combo = K.layers.Dropout(0.5)(combo)
combo = K.layers.Dense(64, activation='relu')(combo)
output = K.layers.Dense(1, activation='sigmoid')(combo)

model = K.Model(inputs=(title_input, abstract_input), outputs=output)
model.summary()

# model.py continued
import tensorflow_datasets as tfds
vocabfile = 'pubmed_vocabulary.txt'
titlefile = 'pubmed_titles.txt'
abstractfile = 'pubmed_abstracts.txt'
journalfile = 'pubmed_journals.txt'

filter_journal_titles = [
    'Ophthalmology',
    'Am J Ophthalmol',
    'Invest Ophthalmol Vis Sci',
    'Br J Ophthalmol',
    'JAMA Ophthalmol',
    'Arch Ophthalmol',
    'Retina',
    'Prog Retin Eye Res',
    'J Cataract Refract Surg',
    'J Vis',
    'Exp Eye Res',
    'Cornea',
    'Acta Ophthalmol',
    'Eye (Lond)',
    'J Refract Surg',
]

print('loading vocabulary')
with open(vocabfile, 'r', encoding='utf-8', errors='surrogateescape') as f:
    vocabulary = []
    for word in f:
        vocabulary.append(word.strip())

tokenizer = tfds.features.text.Tokenizer()
encoder = tfds.features.text.TokenTextEncoder(vocabulary, tokenizer=tokenizer)

def encode_text(text1, text2):
    return encoder.encode(text1.numpy()), encoder.encode(text2.numpy())

def encode_journal(journal):
    if journal in filter_journal_titles:
        return 1
    else:
        return 0

# model.py continued
def encode_text_map_fn(text1, text2):
    return tf.py_function(encode_text, inp=[text1, text2],
                          Tout=(tf.int64, tf.int64))

def encode_journal_map_fn(journal):
    return tf.py_function(encode_journal, inp=[journal], Tout=tf.int64)

# build the tf.data dataset from respective text files
title_dataset = tf.data.TextLineDataset(titlefile)
abstract_dataset = tf.data.TextLineDataset(abstractfile)
journal_dataset = tf.data.TextLineDataset(journalfile)
input_dataset = tf.data.Dataset.zip((title_dataset, abstract_dataset))

# apply transformation functions and combine
input_dataset = input_dataset.map(encode_text_map_fn)
journal_dataset = journal_dataset.map(encode_journal_map_fn)
total_dataset = tf.data.Dataset.zip((input_dataset, journal_dataset))

# model.py continued
# split into training, stop-training, validation, and holdout set
# first 15,000 articles will be held out for final evaluation
test_dataset = total_dataset.take(15000)
test_dataset = test_dataset.padded_batch(
    100,
    padded_shapes=(([72], [200]), ([])))

# model.py continued
metrics = [K.metrics.Recall(name='sens'),
           K.metrics.Precision(name='prec'),
           K.metrics.AUC(name='auc'),
           K.metrics.BinaryAccuracy(name='acc')]
callbacks = [K.callbacks.EarlyStopping(patience=1,
                                       verbose=1, restore_best_weights=True)]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
model.load_weights(keras_model_path)
model.evaluate(test_dataset)
