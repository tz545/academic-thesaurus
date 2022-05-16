import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class Vocabulary:

	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.num_words = 0

	def add_word(self, word):
		if word not in self.word2index:
			# First entry of word into vocabulary
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			# Word exists; increase word count
			self.word2count[word] += 1
			
	def add_wordlist(self, wordlist):
		for word in wordlist:
			self.add_word(word)

	def to_word(self, index):
		if index != -1:
			return self.index2word[index]
		else:
			return '<unk>'

	def to_index(self, word):
		if word in self.word2index:
			return self.word2index[word]
		else:
			return -1

	def to_count(self, word):
		return self.word2count[word]


	def prune_vocab(self, min_word_frequency):
		for word in self.word2index:
			if self.word2count[word] < min_word_frequency:
				index = self.word2index[word]
				del self.word2index[word]
				del self.index2word[index]
				del self.word2count[word]
				self.num_words -= 1


class CBOWDataset(Dataset):

	def __init__(self, dataroot, vocab, window_size):
		self.dataroot = dataroot
		self.vocab = vocab
		self.window_size = window_size
		self.samples = []
		self._init_dataset()

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]

	def _init_dataset(self):
		
		for textfile in os.listdir(self.dataroot):
			with open(os.path.join(self.dataroot, textfile)) as f:
				data = f.readline().strip()
				data = data.split(" ")

			for i in range(self.window_size, len(data)-self.window_size):
				slider = [data[i-j] for j in range(self.window_size, 0, -1)] + [data[i+j] for j in range(1, self.window_size+1)] 
				# print(slider)
				slider = [self.vocab.to_index(x) for x in slider]
				self.samples.append((torch.tensor(slider, dtype=torch.long), self.vocab.to_index(data[i])))


class CBOW_Model(nn.Module):

   def __init__(self, vocab_size, embedding_dim):
       super(CBOW_Model, self).__init__()

       self.embeddings = nn.Embedding(vocab_size, embedding_dim)
       self.linear1 = nn.Linear(embedding_dim, 128)
       self.linear2 = nn.Linear(128, vocab_size)

   def forward(self, inputs):
       # embeds = sum(self.embeddings(inputs)).view(1,-1)
       embeds = self.embeddings(inputs)
       embeds = embeds.mean(axis=1)
       out = F.relu(self.linear1(embeds))
       out = self.linear2(out)
       return out




if __name__ == "__main__":

	datapath = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data")

	v = Vocabulary("test")
	for textfile in os.listdir(datapath):
		with open(os.path.join(datapath, textfile)) as f:
			data = f.readline()
			data = data.split(" ")
			v.add_wordlist(data)
	print(v.num_words)

	dataset = CBOWDataset(datapath, v, 5)

	print(dataset.__len__())
	# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
	# for i, batch in enumerate(dataloader):
	# 	cbow = batch[0]
	# 	word = batch[1]

	# 	print(cbow)
	# 	print(word)

		# for j in range(len(cbow)):
		# 	print([v.to_word(x.item()) for x in cbow[j]], "\t", v.to_word(word[j].item()))
