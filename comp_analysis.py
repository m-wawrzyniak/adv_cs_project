import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tokenedtext_class as tkn
import corpus_class as corp
import freq_analysis as freq_a

'''
This file contains functions for comparative analysis between many tokened objects or within Corpus.
It also contains function allowing for plotting results of these analyses.
Functions:
    - get_tf_idf_batch()
    - plot_tf_idf_matrix()
    - cosine_similarity()
    - plot_cosine_similarity_heatmap()
'''


def get_tf_idf_batch(term_list, obj_tok):
    """
    Computes TF-IDF for a list of terms and either a list of TokenedText objects or a Corpus object,
    return matrix of TF-IDF values as DataFrame.
    :param term_list: (list) List of terms (str) to calculate TF-IDF for.
    :param obj_tok: (Corpus) or (list) of (TokenedText),
    :return: tf_idf_df (DataFrame) of TF-IDF values, where rows are terms and columns are different texts.
    """

    # Different handling of list of TokenedTexts or Corpus.
    if isinstance(obj_tok, list):  # If we got list of TokenedText
        # Validate that obj_tok is a list of TokenedText objects
        for obj in obj_tok:
            if not isinstance(obj, tkn.TokenedText):
                raise TypeError('obj_tok must be of type list of TokenedText or Corpus')

        # Create DataFrame for term counts from nested dictionary
        doc_list = [tok_txt.name for tok_txt in obj_tok]  # Names of files, for column labels
        # term_counts = {<doc_name> : {<term> : <count of term>}}
        term_counts = {doc: {
            term: tok_txt.counts.get(term, 0) for term in term_list  # if no term is found, then count = 0
        } for doc, tok_txt in zip(doc_list, obj_tok)}

        term_df = pd.DataFrame(term_counts).fillna(0)  # If any matrix cell is missing, fill it with 0.

        # Total word counts for each document - pandas Series dataframe
        n_words = pd.Series({doc: tok_txt.n_words for doc, tok_txt in zip(doc_list, obj_tok)})

    elif isinstance(obj_tok, corp.Corpus):  # If we got Corpus
        doc_list = obj_tok.txt_names  # Names of files, for column labels
        # Same as above, but now by accessing corpus texts
        term_counts = {doc: {
            term: obj_tok.corpus_txts[doc].counts.get(term, 0) for term in term_list
        } for doc in doc_list}

        term_df = pd.DataFrame(term_counts).fillna(0)

        # Total word counts for each document
        n_words = pd.Series(obj_tok.n_words, index=doc_list)

    else:
        raise TypeError('obj_tok must be of type list of TokenedText or Corpus')

    # Now we got :
    # term_df with shape (n_terms, n_files) where we got counts for individual terms
    # n_words with shape (n_files, ) where we got total num of words in individual texts

    # Calculate term frequencies (TF): tf = term_df / n_words
    tfs = term_df.div(n_words, axis=1)

    # Calculate document frequencies (DF): In how many documents specific term has occurred
    df = (term_df > 0).sum(axis=1)

    # Calculate inverse document frequencies (IDF)
    n_txt = len(doc_list)
    idf = pd.Series(index=term_list, dtype=float)  # predefine pd.Series for storing IDFs
    idf[df > 0] = np.log(n_txt / df[df > 0])  # For terms which occured at least once within collection of texts
    idf[df == 0] = -np.inf  # Assign -inf for terms that don't appear in any text

    # Calculate TF-IDF : tf-idf = tf * idf
    tf_idf_df = tfs.multiply(idf, axis=0)
    return tf_idf_df


def plot_tf_idf_matrix(tf_idf_df):
    """
    Plots a heatmap of TF-IDF values based on TF-IDF matrix returned from get_tf_idf_batch().
    :param tf_idf_df: (DataFrame) TF-IDF matrix with rows and columns labels.
    :return:
    """
    # Extract values for the heatmap
    tf_idf_matrix = tf_idf_df.values
    term_list = tf_idf_df.index.tolist()  # Terms are the rows
    doc_list = tf_idf_df.columns.tolist()  # Documents are the columns
    mask_nan_vals = np.isnan(tf_idf_matrix)  # Creating mask for NaN values within the matrix

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(  # using Seaborn heatmap
        tf_idf_matrix,
        xticklabels=doc_list,
        yticklabels=term_list,
        cmap="YlOrBr",
        cbar=True,
        annot=True,  # Show the TF-IDF value in the cell
        fmt=".4g",  # Up to 4 significant digits
        linewidths=0.5,  # Space between cells
        ax=ax,
        mask=mask_nan_vals  # Do not show cells with NaN values
    )

    # Fill the cells where there was NaN value.
    for i, term in enumerate(term_list):
        for j, doc in enumerate(doc_list):
            if mask_nan_vals[i, j]:
                ax.text(
                    j + 0.5,  # x-coordinate
                    i + 0.5,  # y-coordinate
                    'NaN',  # Text to show
                    ha='center', va='center', fontsize=10, color='black'
                )

    # Labels, ticks and titles
    ax.set_xlabel("Documents", fontsize=12)
    ax.set_ylabel("Terms", fontsize=12)
    ax.set_title("TF-IDF Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def cosine_similarity(toktxt_a: tkn.TokenedText, toktxt_b: tkn.TokenedText):
    """
    Calculate cosine similarity between two texts.
    :param toktxt_a: (TokenedText)
    :param toktxt_b: (TokenedText)
    :return: (float) Cosine similarity between toktxt_a and toktxt_b
    """
    # Access counts dictionaries
    cnt_a = toktxt_a.counts
    cnt_b = toktxt_b.counts

    # Create set of terms used in union in both texts
    terms_union = set(cnt_a.keys()) | set(cnt_b.keys())

    # Create a vectors spanned on terms_union space
    vec_a = np.array(
        [(cnt_a[term] if term in cnt_a else 0)  # each value within vector is count of specific term
         for term in terms_union]  # and terms are in the same order for both text_a vector and text_b vector
    )
    vec_b = np.array(
        [(cnt_b[term] if term in cnt_b else 0)
         for term in terms_union]
    )

    dot_prod = np.dot(vec_a, vec_b)  # simple dot product on both vectors
    abs_a = np.linalg.norm(vec_a)  # length of both vectors in terms_union space
    abs_b = np.linalg.norm(vec_b)

    return dot_prod / (abs_a * abs_b)  # calculated cosine similarity


def plot_cos_similarity_heatmap(csim_df):
    """
    Plots a heatmap of cosine similarity based on matrix returned from cos_similarity_matrix() method
    in Corpus class.
    :param csim_df: (DataFrame) Cosine similarity matrix with labeled rows and columns.
    :return: (Figure) Plotted heatmap.
    """

    # Creating heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        csim_df,
        annot=True,  # Show values in cells
        fmt=".3f",   # Up to three decimal places
        cmap="YlOrBr",
        cbar=True,
        linewidths=0.5,  # Space between cells
        ax=ax
    )

    # Labels, ticks and titles
    ax.set_xlabel("Texts", fontsize=12)
    ax.set_ylabel("Texts", fontsize=12)
    ax.set_title("Cosine Similarity Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig
