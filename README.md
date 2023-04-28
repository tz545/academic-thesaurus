# academic-thesaurus
Uses word2vec to generate synonym recommendations for specific academic subfields. 

## The Problem

The language used in scientific writing is more precise than in everyday use. In mathematics, for example, words such as "continuous", "real", "stable" have very specific meanings. Even across different scientific subfields, the same English word may carry different meanings, e.g. "field" means entirely different things in quantum physics vs geophysics. When writing scientific articles or reports, standard thesauruses are often of limited use.   

## The Solution

This project draws on the corpus of scientific literature available for download on the preprint server arXiv. It enables the user to download and parse all articles related to a user-inputted keyword, which have a LaTeX file available for download. It then trains the `word2vec` model (implemented from scratch using PyTorch) on the downloaded text corpus. From the custom-trained embeddings, the user is then able to search for words which are semantically close, within the context of the particular academic subfield. 
