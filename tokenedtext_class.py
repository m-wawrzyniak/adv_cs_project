import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import Counter

'''
This file defines TokenedText class, datatype meant to store tokenized content of given *.txt file.
'''


# These are required for tokenize to work
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")


class TokenedText:
    """
    A class for representing *.txt files in properly cleaned and tokenized datatype.

    Attributes:
        name (str) : name of *.txt file on which TokenedText is based
        tokens (list) : List of unique tokens (str) within the text.
        n_words (int) : Number of unique tokens within the text.
        counts (dict) : Tokens (str) as keys and their respective counts (int) as values.
    """

    def __init__(self, txt_path: str, name='_'):
        """
        Constructor for TokenedText class.
        :param txt_path: (str) Path to *.txt file.
        :param name: (str) Identification name of TokenedText
        """
        self.name = name

        raw_content = self.load_txt(txt_path)  # loading content of the file
        content = self.clean_char(raw_content)  # using RegEx to clean the content

        self.tokens = self.tokenize(content)  # tokenizing clean content
        self.n_words = len(self.tokens)  # number of words within the text

        self.counts = self.words_count()  # counting occurrences of unique tokens

    def __str__(self):
        """
        String representation for TokenedText class. Returns (str) of occurring terms and their counts in table format.
        :return: (str)
        """
        max_wrd_len = max(len(wrd) for wrd, _ in self.counts.items())  # longest word, to adjust table size
        print_out = ["Id:| Word:   | Count: |"]  # predefining returned list with pieces of string
        for i, (wrd, cnt) in enumerate(self.counts.items()):
            # Appending to return list: id adjusted to left, token adjusted to left, count adjusted to right
            print_out.append(f"{(str(i + 1) + '.').ljust(3)} {wrd.ljust(max_wrd_len)}; {str(cnt).rjust(2)}")
        return "\n".join(print_out)  # Join the list with newline

    @staticmethod
    def load_txt(path: str):
        """
        Loading *.txt files into single string.
        :param path: (str) Path to *.txt file
        :return: (str) *.txt file converted to string.
        """
        # Some assertion that txt is ok
        file = open(path, 'r', encoding='utf-8')
        content = file.read()

        return content

    @staticmethod
    def clean_char(txt: str):
        """
        Cleaning contents of string containing text. Drops numbers and punctuation, leaving only letters and whitespaces
        :param txt: (str) Raw contents of text file.
        :return: (str) Cleaned contents of text file.
        """
        txt = re.sub("[^a-zA-Z\s]", "", txt)  # Substitute everything apart from a-z, A-Z and whitespaces with blank
        txt = txt.lower()  # Change upper to lower case
        return txt

    @staticmethod
    def tokenize(txt: str):
        """
        Converts single cleaned string with contents of text file into list of individual words called tokens.
        :param txt: (str) Cleaned contents of text file.
        :return: (list) List of individual words in the file (str)
        """
        tokens = word_tokenize(txt)  # Utilize nltk word_tokenize

        # Drop common stop words using nltk - this lib has pretty conservative set of stop-words
        stop_words = set(stopwords.words("english"))
        tokens = [tk for tk in tokens if tk not in stop_words]

        # Drop any words shorter than 3 characters to increase quality of some text transcriptions
        tokens = [tk for tk in tokens if len(tk) >= 3]
        return tokens

    def words_count(self):
        """
        Count occurrences of individual tokens within TokenedText. Zip them in a dictionary.
        :return: (dict) Keys are tokens, values are counts.
        """
        txt_count = Counter(self.tokens)  # Utilizing Counter function from collections
        sorted_count = {k: v for k, v in sorted(txt_count.items(),
                                                key=lambda item: item[1],  # Sorting dictionary with respect to count
                                                reverse=True)}  # Reversing the order, so the highest counts are in front
        return sorted_count
