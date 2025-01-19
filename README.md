# Project: Word Frequency Analysis of Texts
**Made by Mateusz Wawrzyniak**

This project objective was to create a word frequency analyzer using Python. Specific tasks completed include:

### 1. Data Collection
- Texts were manually collected in `.txt` format and are located in the `exemplar_texts` directory.
- Analyzed texts include:
  - Plato's *The Republic*
  - Issues of *The New York Times*
  - Works by Kafka, Orwell, Hesse, Dylan
  - Wikipedia articles
  - Scientific books by Harari, Sapolsky, and Dawkins

Each text corpus was analyzed individually in `.ipynb` files. Start with `a_Plato_Republic_analysis.ipynb` and proceed alphabetically.

### 2. Data Preprocessing
- A custom class, `TokenedText` (in `tokenedtext_class.py`), was created for cleaning, converting, and tokenizing text data.

### 3. Word Frequency Analysis
- Functions for generating bar graphs and word clouds are in `freq_analysis.py`, designed to operate on `TokenedText` and `Corpus` objects.

### 4. Comparative Analysis
- A `Corpus` class (in `corpus_class.py`) was introduced to analyze similarities and unique words using TF-IDF and cosine similarity (`comp_analysis.py`).

### 5. Data Visualization
- Plots like bar graphs, heatmaps, and word clouds are created in `freq_analysis.py` and `comp_analysis.py`.

### 6. Conclusion and Evaluation
- Evaluations and conclusions are documented for the first two corpora in the notebooks.

### **Reconstructing Virtual Environment and Jupyter Kernel**
1. Create a directory where project and venv will be stored
2. Using CMD, go to the created directory: cd <path_to_directory>
3. CMD > git clone https://github.com/m-wawrzyniak/adv_cs_text_analysis
4. cd adv_cs_text_analysis
5. python -m venv txt_an_venv
6. venv\Scripts\activate
7. pip install -r requirements.txt
8. Leave the virtual environment
9. python -m ipykernel install --user --name=txt_an_venv --display-name "Txt analysis kernel"
10. In the directory, run jupyter notebook

**Enjoy the project!**

