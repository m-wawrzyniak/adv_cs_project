import corpus_class as dc
import tokenedtext_class as prep
import freq_analysis as freq_a
import comp_analysis as comp_a
import result_visualisation as re_vis

import matplotlib.pyplot as plt

corpus_novels = dc.Corpus("../exemplar_texts/novels_poems")

# Basic corpus analysis:
# Number of distinct tokens, number of word count, number of texts, text names
novels_info = corpus_novels.get_basic_info()
print(novels_info)

# Most common words in whole corpus in bar plot
novels_common = freq_a.create_bar_count(corpus_novels, n=25)
plt.show()

# Most common words in whole corpus in wordcloud
novels_wordcloud = freq_a.create_word_cloud(corpus_novels)
plt.show()

# Comparative analysis:
# Cosine similarity pair-wise across corpus - in table
novels_csim = corpus_novels.cos_similarity_matrix()
novels_csim_fig = comp_a.plot_cos_similarity_heatmap(novels_csim)
plt.show()

# TD-IDF for 15 most common words in corpus
novels_rand_toks = corpus_novels.get_random_tokens(n=10, seed=293)
novels_idf_tf = comp_a.get_tf_idf_batch(novels_rand_toks, corpus_novels)
novels_idf_tf_fig = comp_a.plot_tf_idf_matrix(novels_idf_tf)
plt.show()

# Specific analysis:
# None
