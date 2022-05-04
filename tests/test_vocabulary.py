from AcademicThesaurus.word2vec_cbow import Vocabulary

def test_add_word_creates_new_entry():
	voc = Vocabulary("test")
	voc.add_word("firstword")
	assert voc.name == "test"
	assert voc.word2index == {"firstword":0}
	assert voc.word2count == {"firstword":1}
	assert voc.index2word == {0:"firstword"}
	assert voc.num_words == 1

def test_add_word_increments_entry():
	voc = Vocabulary("test")
	voc.add_word("firstword")
	voc.add_word("firstword")
	assert voc.name == "test"
	assert voc.word2index == {"firstword":0}
	assert voc.word2count == {"firstword":2}
	assert voc.index2word == {0:"firstword"}
	assert voc.num_words == 1

def test_add_word_adds_new_entry():
	voc = Vocabulary("test")
	voc.add_word("firstword")
	voc.add_word("secondword")
	assert voc.name == "test"
	assert voc.word2index == {"firstword":0, "secondword":1}
	assert voc.word2count == {"firstword":1, "secondword":1}
	assert voc.index2word == {0:"firstword", 1:"secondword"}
	assert voc.num_words == 2

def test_add_wordlist_to_new():
	voc = Vocabulary("test")
	wordlist = ["firstword", "secondword"]
	voc.add_wordlist(wordlist)
	assert voc.name == "test"
	assert voc.word2index == {"firstword":0, "secondword":1}
	assert voc.word2count == {"firstword":1, "secondword":1}
	assert voc.index2word == {0:"firstword", 1:"secondword"}
	assert voc.num_words == 2

def test_add_wordlist_to_existing():
	voc = Vocabulary("test")
	voc.add_word("firstword")
	voc.add_word("firstword")
	wordlist = ["firstword", "secondword"]
	voc.add_wordlist(wordlist)
	assert voc.name == "test"
	assert voc.word2index == {"firstword":0, "secondword":1}
	assert voc.word2count == {"firstword":3, "secondword":1}
	assert voc.index2word == {0:"firstword", 1:"secondword"}
	assert voc.num_words == 2

def test_retriever_functions():
	voc = Vocabulary("test")
	voc.add_word("firstword")
	voc.add_word("firstword")
	wordlist = ["firstword", "secondword"]
	voc.add_wordlist(wordlist)
	assert voc.to_word(0) == "firstword"
	assert voc.to_word(1) == "secondword"
	assert voc.to_index("firstword") == 0
	assert voc.to_index("secondword") == 1
	assert voc.to_count("firstword") == 3
	assert voc.to_count("secondword") == 1