import os
import random
import pandas as pd

import tokenedtext_class as tkn
import comp_analysis as comp_a

# TODO: DONE
'''
This file defines Corpus class, which will be default object used for interfile comparative analysis.
'''


class Corpus:
    """
    A class representing corpus of *.txt files which will be tokenized and analysed.

    Attributes:
        name (str) : name of folder on which corpus is based
        corpus_txts (dict) :
            <keys> (str) : *.txt file names,
            <values> (TokenedText) : cleaned and tokenized text content
        n_txt (int) : number of text in the corpus
        txt_names (list) : *.txt file names (str)
        tokens (list) : unique words occurring in the corpus (str)
        counts (dict) :
            <keys> (str) : token within corpus
            <values> (int) : number of token occurrences within the corpus
    """
    def __init__(self, folder_path: str):
        """
        The constructor for Corpus class.
        Parameters:
        :param folder_path (str): Path to folder with all *.txt files which will be part of the corpus
        """
        self.name = folder_path.split('\\')[-1]  # corpus name is folder name
        self.corpus_txts, self.n_txt, self.txt_names = self.create_corpus_dict(folder_path)
        self.tokens = self.corpus_tokenize()
        self.counts, self.n_words = self.corpus_words_count()
        print(f'Corpus {self.name} has been created')

    def __str__(self):
        """
        String representation for Corpus class.
        :return self.name (str): Name attribute
        """
        return self.name

    @staticmethod
    def create_corpus_dict(folder_path: str):
        """
        Creates 'corpus_txts' attribute for Corpus class.
        :param folder_path: (str) Path to folder on which Corpus will be based
        :return corpus_dict: (dict) Dictionary with individual *.txt files names as keys and TokenedText as values
        :return n_txt: (int) Number of *.txt files included in Corpus
        :return txt_names: (list) List of *.txt files names (str)
        """
        # Predefining returning data
        corpus_dict = {}
        n_txt = 0
        txt_names = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):  # ensure to take only *.txt files
                n_txt += 1  # count the texts
                txt_names.append(file_name)
                print(f'Processing {file_name}...')  # terminal output to make sure we got all texts
                txt_path = os.path.join(folder_path, file_name)  # joining full path to individual *.txt file
                corpus_dict[file_name] = tkn.TokenedText(txt_path, name=file_name)  # creating TokenedText for this file
        return corpus_dict, n_txt, txt_names

    def corpus_tokenize(self):
        """
        Creates a list of all tokens which occur in files included in Corpus.
        :return: (list) of tokens (str)
        """
        tok_set = set()
        for toks in self.corpus_txts.values():  # for each TokenedText
            tok_set = tok_set | set(toks.tokens)  # we extract tokens and create union set with tokens got so far
        return sorted(list(tok_set))  # convert to list and sort - sorting allows for replicating random picks with seed

    def corpus_words_count(self):
        """
        Counts all occurrences of tokens occurring within the Corpus. Also counts total number of words within Corpus.
        :return sorted_count: (dict) Tokens (str) for keys and occurrences of token (int) for values.
        :return num_words: (int) Total number of words within the Corpus.
        """
        counts = {k: 0 for k in self.tokens}  # predefining dict with tokens in place
        num_words = 0
        for tok_txt in self.corpus_txts.values():  # for each TokenedText in Corpus
            for t, cnt in tok_txt.counts.items():  # for each term and its count in TokenedText
                counts[t] += cnt  # add number of occurrences to total number within Corpus
                num_words += cnt

        # Sort the dict for easier handling
        sorted_count = {k: v for k, v in sorted(counts.items(),  # sort counts items..
                                                key=lambda item: item[1],  # based on value of the item (count value)..
                                                reverse=True)}  # and reverse the order, so highest are in front.
        return sorted_count, num_words

    def get_basic_info(self):
        """
        Prepares basic information concerning Corpus.
        :return: (str)
        """
        intro = f'Basic info concerning {self.name} corpus:'
        txt_count = f'Number of text files: {self.n_txt}'
        distinct_toks = f'Number of distinct tokens: {len(self.tokens)}'
        word_count = f'Total number of words: {self.n_words}'

        return '\n'.join((intro, txt_count, distinct_toks, word_count))

    def cos_similarity_matrix(self):
        """
        Prepares cosine similarity matrix pair-wise between individual texts within Corpus.
        :return: (DataFrame) Cosine similarity matrix
        """
        csim_df = pd.DataFrame(index=self.txt_names, columns=self.txt_names, dtype=float)  # <n_txt x n_txt> Dataframe
        for i, txt_a in enumerate(self.txt_names):  # for each text name...
            if i+1 >= len(self.txt_names):  # if there is no more texts, then drop it
                continue
            else:  # else, there are some texts left...
                for txt_b in self.txt_names[i+1:]:  # grab the next one...
                    c_sim = comp_a.cosine_similarity(self.corpus_txts[txt_a], self.corpus_txts[txt_b])  # compute...
                    csim_df.at[txt_a, txt_b] = c_sim
                    csim_df.at[txt_b, txt_a] = c_sim  # write into matrix, symmetrically (property of cosine sim)
        csim_df.fillna(1.0, inplace=True)  # cells which are left with no value are on the diagonal, so they have val=1

        return csim_df

    def get_random_tokens(self, n=10, seed=0):
        """
        Helper function to get n random tokens from Corpus based on seed.
        :param n: (int) Number of return tokens
        :param seed: (int) Seed for random generator
        :return: (list) of tokens (str)
        """
        random.seed(seed)
        # making sure im not asking for more tokens than there are within Corpus
        return random.sample(self.tokens, min(n, len(self.tokens)))
