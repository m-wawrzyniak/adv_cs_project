import corpus_class as corp
import tokenedtext_class as tkn
import freq_analysis as freq_a
import comp_analysis as comp_a

import matplotlib.pyplot as plt

'''
This file contains functions for special cases of result plotting within different Corpora
'''


def change_over_time_plato(corpus: corp.Corpus, terms):
    """
    Plots terms count over successive books within Plato's Republic Corpus.
    :param corpus: (Corpus) Plato's Republic Corpus.
    :param terms: (list) List of terms (str), which will be plotted against successive books.
    :return: (Figure) Plotted figure.
    """
    if corpus.name != 'plato_republic':  # Making sure we got plato_republic corpus
        raise ValueError(f"Invalid Corpus provided. Got: {corpus.name}, Expected: 'plato_republic'")

    # Predefining books indices and figure
    books = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    fig, ax = plt.subplots(figsize=(10, 6))

    for term in terms:  # For each term in term list
        valid_books = []  # Books found in the Corpus
        term_counts = []  # Counts of this term
        for b in books:
            for txt_name, tok_txt in corpus.corpus_txts.items():
                if b in txt_name:  # Making sure we got that book indices within the Corpus
                    valid_books.append(int(b))
                    if term in tok_txt.counts:
                        term_counts.append(tok_txt.counts[term])  # Appending count of specific term from current book
                    else:
                        term_counts.append(0)  # If no token found in this book, append 0

        ax.plot(valid_books, term_counts, marker='o', label=f"Cnt of '{term}'")  # Plotting current term

    # Labels, ticks and titles
    ax.set_xticks([int(b) for b in books])
    ax.set_xticklabels(books)
    ax.set_xlabel("Book", fontsize=12)
    ax.set_ylabel("Term Count", fontsize=12)
    ax.set_title(f"Trend of {terms} over books in {corpus.name}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)  # Grid, for better visibility
    ax.legend()

    return fig


def change_over_time_times(corpus: corp.Corpus, terms):
    """
    Plots terms count over successive years between 1938 and 1946 within The New York Times Corpus.
    :param corpus: (Corpus) The New York Times Corpus.
    :param terms:  (list) List of terms (str), which will be plotted against successive years.
    :return:  (Figure) Plotted figure.
    """
    if corpus.name != 'the_times':  # Making sure we got the_times corpus
        raise ValueError(f"Invalid Corpus provided. Got: {corpus.name}, Expected: 'the_times'")

    # Predefining years range and figure
    years = range(38, 47)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Same as in function change_over_time_plato
    for term in terms:
        valid_years = []
        term_counts = []
        for y in years:
            for txt_name, tok_txt in corpus.corpus_txts.items():
                if str(y) in txt_name:
                    valid_years.append(y)
                    if term in tok_txt.counts:
                        term_counts.append(tok_txt.counts[term])
                    else:
                        term_counts.append(0)
        ax.plot(valid_years, term_counts, marker='o', label=f"Count of '{term}'")

    # Labels, ticks and titles
    ax.set_xticks(years)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Term Count", fontsize=12)
    ax.set_title(f"Trend of {terms} over years in {corpus.name}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()

    return fig
