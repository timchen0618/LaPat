import csv

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3

class Vocab:
    def __init__(self):
        self.word2index = {"<UNK>": 0, "<PAD>":1, "<SOS>":2, "<EOS>":3}
        self.word2count = {}
        self.index2word = {0: "<UNK>", 1: "<PAD>", 2:"<SOS>", 3:"<EOS>"}
        self.n_words = 4  # Count SOS and EOS
        self.vocab_size_threshold = 50004
        self.sos = "<SOS>"
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        self.unk = "<UNK>"

    def wordCounter(self, dataset):
        with open(dataset, 'r') as tsv_in:
            tsv_reader = csv.reader(tsv_in, delimiter='\t')

            for line in tsv_reader:
                for word in line[0].split():
                    if word not in self.word2count:
                        self.word2count[word] = 1
                    else:
                        self.word2count[word] += 1

                for word in line[2].split():
                    if word not in self.word2count:
                        self.word2count[word] = 1
                    else:
                        self.word2count[word] += 1

        self.word2count = sorted(self.word2count.items(), key=lambda d: d[1], reverse=True)
        self.word2count = {tup[0]:tup[1] for tup in self.word2count}


    def addWord(self, freq_threshold=1):
        for word in self.word2count.keys():
            if (self.word2count[word] >= freq_threshold) and (word not in self.word2index) and (self.n_words < self.vocab_size_threshold):
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1


    def word2id(self, word):
        if word not in self.word2index:
          return self.word2index[self.unk]
        return self.word2index[word]


    def id2word(self, word_id):
        if word_id not in self.index2word:
          raise ValueError('Id not found in vocab: %d' % word_id)
        return self.index2word[word_id]



def input2ids(input_words, vocab):
    ids = []
    unk_id = vocab.word2id(UNK_token)

    for w in input_words.split():
        i = vocab.word2id(w)
        if i == unk_id:    # If w is OOV
            ids.append(unk_id)    # Map to the UNK token id
        else:
            ids.append(i)

    return ids


def latent2ids(latent_words, vocab, oovs=[]):
    ids = []
    unk_id = vocab.word2id(UNK_token)

    for w in latent_words.split():
        i = vocab.word2id(w)
        if i == unk_id:    # If w is OOV
            if w not in oovs:    # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)    # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.n_words + oov_num)    # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)

    return ids, oovs


def target2ids(target_words, vocab, latent_oovs):
    ids = []
    unk_id = vocab.word2id(UNK_token)

    for w in target_words.split():
        i = vocab.word2id(w)
        if i == unk_id:    # If w is an OOV word
            if w in latent_oovs:    # If w is an in-article OOV
                vocab_idx = vocab.n_words + latent_oovs.index(w)    # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:    # If w is an out-of-article OOV
                ids.append(unk_id)    # Map to the UNK token id
        else:
            ids.append(i)

    return ids
