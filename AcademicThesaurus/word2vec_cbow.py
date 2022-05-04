import os
import torch
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
		return self.index2word[index]

	def to_index(self, word):
		return self.word2index[word]

	def to_count(self, index):
		return self.word2count[word]


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
				data = f.readline()
				data = data.split(" ")

			for i in range(self.window_size, len(data)-self.window_size):
				slider = [data[i-j] for j in range(self.window_size, 0, -1)] + [data[i+j] for j in range(1, self.window_size+1)] 
				# print(slider)
				slider = [self.vocab.to_index(x) for x in slider]
				self.samples.append((torch.Tensor(slider), self.vocab.to_index(data[i])))




if __name__ == "__main__":

	datapath = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data")

	v = Vocabulary("test")
	for textfile in os.listdir(datapath):
		with open(os.path.join(datapath, textfile)) as f:
			data = f.readline()
			data = data.split(" ")
			v.add_wordlist(data)

	dataset = CBOWDataset(datapath, v, 5)
	dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
	for i, batch in enumerate(dataloader):
		cbow = batch[0]
		word = batch[1]

		for j in range(len(cbow)):
			print([v.to_word(x.item()) for x in cbow[j]], "\t", v.to_word(word[j].item()))
