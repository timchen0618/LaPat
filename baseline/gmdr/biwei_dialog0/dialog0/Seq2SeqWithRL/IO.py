from __future__ import unicode_literals
from __future__ import print_function
import torchtext
from collections import defaultdict,Counter
import codecs
from itertools import count
PAD_WORD = '<blank>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'
USE_RL = False

def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__



def get_fields():

    fields = {}
    fields["src"] = torchtext.data.Field(init_token=BOS_WORD,pad_token=PAD_WORD,include_lengths=True)
    fields["tgt"] = torchtext.data.Field(init_token=BOS_WORD,eos_token=EOS_WORD,pad_token=PAD_WORD,include_lengths=True)
    fields["tag"] = torchtext.data.Field(pad_token=PAD_WORD)

    return fields

def load_fields(vocab):
    vocab = dict(vocab)
    fields = get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        if k in fields:
            fields[k].vocab = v
    return fields

def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab
    
def build_vocab(train, vocab):
    fields = train.fields
    vocab = dict(vocab)
    fields["src"].vocab = vocab['src']
    fields["tgt"].vocab = vocab['tgt']
    fields["tag"].vocab = vocab['tag']



# OrderedIterator = torchtext.data.Iterator


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            sofar+=1
            return sofar
    minibatch, size_so_far = [], 0
    for ex in data:
        if "_PAD" not in ex.tgt:
            minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size and len(minibatch)==0:
            minibatch, size_so_far = [], 0
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch

class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)

class InferIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
