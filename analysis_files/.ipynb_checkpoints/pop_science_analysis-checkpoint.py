import corpus_class as dc
import tokenedtext_class as prep
import freq_analysis as freq_a
import comp_analysis as comp_a
import result_visualisation as re_vis

import matplotlib.pyplot as plt

corpus_science = dc.Corpus("../exemplar_texts/pop_science")

# Basic corpus analysis:
# Number of distinct tokens, number of word count, number of texts, text names
science_info = corpus_science.get_basic_info()
print(science_info)

# Most common words in whole corpus in bar plot
science_common = freq_a.create_bar_count(corpus_science, n=25)
plt.show()

# Most common words in whole corpus in wordcloud
science_wordcloud = freq_a.create_word_cloud(corpus_science)
plt.show()

# Comparative analysis:
# Cosine similarity pair-wise across corpus - in table
science_csim = corpus_science.cos_similarity_matrix()
science_csim_fig = comp_a.plot_cos_similarity_heatmap(science_csim)
plt.show()

# TD-IDF for 15 most common words in corpus
science_random_toks = corpus_science.get_random_tokens(n=15, seed=974)
science_tf_idf = comp_a.get_tf_idf_batch(science_random_toks, corpus_science)
science_tf_idf_fig = comp_a.plot_tf_idf_matrix(science_tf_idf)
plt.show()

# Specific analysis:
# None
