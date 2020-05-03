## Using Pubmed data and tf.keras to predict if an ophthalmology paper will make it into a top 15 journal
For [this tutorial](https://benjamintseng.com), I show how to write a simple algorithm using tf.keras to predict if a paper, based solely on its abstract and title, is one published in one of the ophthalmology field's top 15 journals (as gauged by [Google Scholar on the basis of h-index from 2014-2018](https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=med_ophthalmologyoptometry). 
It will be trained and tested on a dataset comprising every English-language ophthalmology-related article with an abstract in [Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/) (the free search engine operated by NIH covering practically all life sciences-related papers) published from 2010 to 2019. 

Beyond using basic Tensorflow features and Keras layers (fully connected layers, word embeddings, recurrent neural networks), it will incorporate some functionality (e.g., multiple input models using tf.keras and tf.data, using the same embedding layer on multiple input sequences, using padded batches with tf.data to incorporate sequence data) that have (by my own assessment) been documented more poorly but which I've used on multiple projects. 

This piece also assumes you know the basics of Python 3 (and Numpy) and have installed [Tensorflow 2](https://www.tensorflow.org/install) (with GPU support), [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), [lxml](https://lxml.de/), and [Tensorflow Datasets](https://www.tensorflow.org/datasets).

For more information (including detailed description of the different parts of the code), refer to the post at [my webpage](https://benjamintseng.com/).
