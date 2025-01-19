import corpus_class as dc
import tokenedtext_class as prep
import freq_analysis as freq_a
import comp_analysis as comp_a
import result_visualisation as re_vis

import matplotlib.pyplot as plt

corpus_the_times = dc.Corpus("../exemplar_texts/the_times")

# Basic corpus analysis:
# Number of distinct tokens, number of word count, number of texts, text names
the_times_info = corpus_the_times.get_basic_info()
print(the_times_info)

# Most common words in whole corpus in bar plot
the_times_common = freq_a.create_bar_count(corpus_the_times, n=25)
plt.show()

# Most common words in whole corpus in wordcloud
the_times_wordcloud = freq_a.create_word_cloud(corpus_the_times)
plt.show()

# Comparative analysis:
# Cosine similarity pair-wise across corpus - in table
the_times_csim = corpus_the_times.cos_similarity_matrix()
the_times_csim_fig = comp_a.plot_cos_similarity_heatmap(the_times_csim)
plt.show()

# TD-IDF for 15 most common words in corpus
times_rand_toks = corpus_the_times.get_random_tokens(n=10, seed=452)
times_toks = ['behooves', 'skylights', 'war', 'polish', 'burma']
times_tf_idf = comp_a.get_tf_idf_batch(times_toks, corpus_the_times)
times_td_idf_fig = comp_a.plot_tf_idf_matrix(times_tf_idf)
plt.show()

# Specific analysis:
# Change of most common words over years in different articles - OK [war, hitler, bonds, poland]
times_change_fig = re_vis.change_over_time_times(corpus_the_times, ['war', 'hitler', 'bonds', 'poland', 'japanese'])
plt.show()
