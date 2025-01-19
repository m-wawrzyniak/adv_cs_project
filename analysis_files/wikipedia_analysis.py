import corpus_class as dc
import tokenedtext_class as prep
import freq_analysis as freq_a
import comp_analysis as comp_a
import result_visualisation as re_vis

import matplotlib.pyplot as plt

corpus_wiki = dc.Corpus("../exemplar_texts/wikipedia_articles")

# Basic corpus analysis:
# Number of distinct tokens, number of word count, number of texts, text names
wiki_info = corpus_wiki.get_basic_info()
print(wiki_info)

# Most common words in whole corpus in bar plot
wiki_common = freq_a.create_bar_count(corpus_wiki, n=25)
plt.show()

# Most common words in whole corpus in wordcloud
wiki_wordcloud = freq_a.create_word_cloud(corpus_wiki)
plt.show()

# Comparative analysis:
# Cosine similarity pair-wise across corpus - in table
wiki_csim = corpus_wiki.cos_similarity_matrix()
wiki_csim_fig = comp_a.plot_cos_similarity_heatmap(wiki_csim)
plt.show()

# TD-IDF for 15 most common words in corpus
wiki_rand_toks = corpus_wiki.get_random_tokens(n=15, seed=582)
wiki_tf_idf = comp_a.get_tf_idf_batch(wiki_rand_toks, corpus_wiki)
wiki_tf_idf_fig = comp_a.plot_tf_idf_matrix(wiki_tf_idf)
plt.show()

# Specific analysis:
# Some graph which will show which articles concern the same topic
