import matplotlib.pyplot as plt
from wordcloud import WordCloud

import corpus_class as corp
import tokenedtext_class as tkn

'''
This file contains functions for frequency analysis within single tokened object.
Functions:
    - create_bar_count()
    - create_word_cloud()
'''


def create_bar_count(obj_tok, n=10):
    """
    Creates bar plot of n most common used words within tokenized object obj_tok.
    :param obj_tok: (TokenedText) or (Corpus) Tokenized object.
    :param n: (int) Number of most common words meant to be on the bar plot.
    :return: (Figure) Bar plot.
    """
    # Different handling of TokenedText or Corpus
    if isinstance(obj_tok, tkn.TokenedText):
        cnts = obj_tok.counts
        name = f'text {obj_tok.name}'
    elif isinstance(obj_tok, corp.Corpus):
        cnts = obj_tok.counts
        name = f'corpus {obj_tok.name}'
    else:
        raise TypeError('obj_tok must be of type TokenedText or Corpus')  # Making sure we got correct obj_tok type

    # Making sure we won't try to print more terms than there are within the tokenized object
    size = len(list(cnts.keys()))
    if n >= size:
        n = size

    # Counts of tokens within obj_tok is always sorted, so just take n first items
    words = list(cnts.keys())[:n]
    cnts = list(cnts.values())[:n]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, cnts, color='skyblue', edgecolor='black')

    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Top {n} most common words in {name}", fontsize=14)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='center', fontsize=8)  # Rotate it a bit, so there is no text overlap
    fig.tight_layout()

    return fig


def create_word_cloud(obj_tok):
    """
    Creates word cloud based on words occurrences within tokenized object obj_tok.
    :param obj_tok: (TokenedText) or (Corpus) Tokenized object.
    :return: (Figure) WordCloud plot.
    """
    # Different handling of TokenedText or Corpus.
    if isinstance(obj_tok, tkn.TokenedText):
        cnts = obj_tok.counts
        name = f'text {obj_tok.name}'
    elif isinstance(obj_tok, corp.Corpus):
        cnts = obj_tok.counts
        name = f'corpus {obj_tok.name}'
    else:
        raise TypeError('obj_tok must be of type TokenedText or Corpus')

    # Utilizing wordcloud library with dictionary of {term: count}
    wordcloud = WordCloud(width=800, height=600, background_color='black').generate_from_frequencies(cnts)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('black')  # Black background
    ax.imshow(wordcloud, interpolation='bilinear')  # Some interpolation to smooth the image edges
    ax.axis('off')
    ax.set_title(f'WordCloud for {name}', color='white')

    return fig

