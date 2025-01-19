import corpus_class as dc
import tokenedtext_class as prep
import freq_analysis as freq_a
import comp_analysis as comp_a
import result_visualisation as re_vis

import matplotlib.pyplot as plt

plato_corpus = dc.Corpus('../exemplar_texts/plato_republic')

# Basic corpus analysis:
# Number of distinct tokens, number of word count, number of texts - OK
plato_info = plato_corpus.get_basic_info()
print(plato_info)

# Most common words in whole corpus in bar plot - OK
plato_common = freq_a.create_bar_count(plato_corpus, n=20)
plt.show()

# Most common words in whole corpus in wordcloud - OK
plato_wordcloud = freq_a.create_word_cloud(plato_corpus)
plt.show()

# Comparative analysis:
# Cosine similarity pair-wise across corpus - in table - OK
plato_csim = plato_corpus.cos_similarity_matrix()
plato_csim_fig = comp_a.plot_cos_similarity_heatmap(plato_csim)
plt.show()

# TD-IDF for 15 most common words in corpus - OK
plato_rand_toks = plato_corpus.get_random_tokens(n=15, seed=123)
plato_tf_idf = comp_a.get_tf_idf_batch(plato_rand_toks, plato_corpus)
plato_td_idf_fig = comp_a.plot_tf_idf_matrix(plato_tf_idf)
plt.show()

# Find term with highest TD-IDF for each text and show them - LEFT

# Specific analysis:
# Changes of most common words across different books - line plot - OK [justice, glaucon, division]
plato_change_fig = re_vis.change_over_time_plato(plato_corpus, ['justice', 'glaucon', 'division'])
plt.show()
