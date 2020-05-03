# parse-pubmed.py
from bs4 import BeautifulSoup, NavigableString
import tensorflow_datasets as tfds
import csv
import collections
import random

# input and output file names
xmlfile = 'pubmed_result.xml'
vocabularyfile = 'pubmed_vocabulary.txt'
titlefile = 'pubmed_titles.txt'
abstractfile = 'pubmed_abstracts.txt'
journalfile = 'pubmed_journals.txt'

# initialize key variables
parsestep = 1000  # number of articles parsed before issuing an interim update
smallestfreq = 25  # smallest word frequency to make it into tracked vocab
maxlength = 200  # word length cutoff for abstracts / titles

vocab = collections.Counter()  # initialize counter (to track vocab)
tokenizer = tfds.features.text.Tokenizer()  # use tensorflow's tokenizer
resultsList = []  # tracks all the abstracts, titles, and values

with open(xmlfile, 'r', encoding='utf-8', errors='surrogateescape') as f:
    line = f.readline()
    article = ''
    articles = 0
    while line:
        # signifies a new article
        if '<PubmedArticle>' in line: 
            article += line[0:line.index('<PubmedArticle>')]
            # track progress
            if articles % parsestep == 0 and articles > 0:
                print('Article:', articles)

            # use BeautifulSoup to read XML
            soup = BeautifulSoup(' '.join(article.split()).replace(
                '> <', '><'), 'xml')

            # extract title
            title = ''
            titletag = soup.find('ArticleTitle')
            title = titletag.get_text() if titletag else None

            # extract abstract
            abstracttag = soup.find('Abstract')
            abstract = ''
            if not abstracttag:
                for tag in soup.find_all('OtherAbstract'):
                    if 'Language' in tag.attrs and tag['Language'] == 'eng':
                        abstracttag = tag
                        break

            if abstracttag:
                if len(abstracttag.contents) == 1:
                    abstract = abstracttag.get_text()
                else:
                    for child in abstracttag.contents:
                        if child.name == 'AbstractText':
                            if 'Label' in child.attrs:
                                if len(abstract) > 1:
                                    abstract += ' ' + child['Label']
                                    abstract += ': ' + child.get_text()
                                else:
                                    abstract += child['Label']
                                    abstract += ': ' + child.get_text()
                            else:
                                abstract += child.get_text()
                        elif isinstance(child, NavigableString):
                            abstract += child.string if child.string else ''

            # extract journal name
            journalname = ''
            journaltag = soup.find('MedlineJournalInfo')
            if journaltag:
                journalname = journaltag.find('MedlineTA').get_text().strip()

            # check if abstract length, title length, journal name make sense
            # if yes, tokenize and truncate, update vocab, and store
            if len(abstract) > 50 and len(title) > 5 and journalname != '':
                # convert title and abstract
                title = title.strip().replace('\n', ' ').lower()
                abstract = abstract.strip().lower()
                journalname = journalname.strip().replace('\n', ' ')

                abstractwords = tokenizer.tokenize(abstract)
                abstractwords = abstractwords[0:maxlength]
                abstract = ' '.join(abstractwords)

                titlewords = tokenizer.tokenize(title)
                titlewords = titlewords[0:maxlength]
                title = ' '.join(titlewords)

                vocab.update(abstractwords)
                vocab.update(titlewords)

                resultsList.append([title, abstract, journalname])
                articles += 1
            else:
                # print errors
                print('Failed, title:', title,
                      ', in journal:', journalname,
                      ', abstract length:', len(abstract))

            article = line[line.index('<PubmedArticle>'):]
            line = f.readline()
        else:
            article = article + line
            line = f.readline()

# parse-pubmed.py continued
# files to store parsed information
g = open(vocabularyfile, 'w', encoding='utf-8', errors='surrogateescape')
h = open(titlefile, 'w', encoding='utf-8', errors='surrogateescape')
i = open(abstractfile, 'w', encoding='utf-8', errors='surrogateescape')
j = open(journalfile, 'w', encoding='utf-8', errors='surrogateescape')

print('writing vocabulary')
for word, num in vocab.most_common():
    if num < smallestfreq:
        break
    g.write(word + '\n')

# shuffle (so train and test aren't purely off of XML order)
print('shuffling results')
# random.seed(100) # uncomment if you want the same shuffle every time
random.shuffle(resultsList)

print('writing results to files')
for row in resultsList:
    title, abstract, journalname = row
    h.write(title+'\n')
    i.write(abstract+'\n')
    j.write(journalname+'\n')

g.close()
h.close()
i.close()
j.close()