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
    fields["tgt"] = torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,pad_token=PAD_WORD,include_lengths=True)
    fields["tag"] = torchtext.data.Field(pad_token=PAD_WORD)
    return fields

def load_fields(vocab):
    vocab = dict(vocab)
    fields = get_fields()
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields

def save_vocab(fields):
    vocab = []
    for k, f in fields.items():
        if 'vocab' in f.__dict__:
            f.vocab.stoi = dict(f.vocab.stoi)
            vocab.append((k, f.vocab))
    return vocab
    
def build_vocab(train, config):
    fields = train.fields
    fields["src"].build_vocab(train, max_size=config['Misc']['src_vocab_size'])
    fields["tgt"].build_vocab(train, max_size=config['Misc']['tgt_vocab_size'])
    fields["tag"].build_vocab(train, max_size=config['Misc']['tag_vocab_size'])
    # `tgt_vocab_size` is ignored when sharing vocabularies
    merged_vocab = merge_vocabs(
        [fields["src"].vocab, fields["tgt"].vocab],
        vocab_size=config['Misc']['src_vocab_size'])
    fields["src"].vocab = merged_vocab
    fields["tgt"].vocab = merged_vocab
    fields["tag"].vocab = merged_vocab
    
def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.
    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[PAD_WORD, BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)

class DialogDataset(torchtext.data.Dataset):


    def __init__(self, src_path, tgt_path, tag_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist
        if not USE_RL:
            print('not use rl')
            with codecs.open(src_path, encoding="utf8",errors='replace') as src_f, \
                codecs.open(tgt_path, encoding="utf8",errors='replace') as tgt_f, \
                codecs.open(tag_path, encoding="utf8",errors='replace') as tag_f:
                examples = []
                for src,tgt,tag in zip(src_f,tgt_f,tag_f):
                    if tag.strip() == "":
                        tag = PAD_WORD
                    examples.append(make_example([src,tgt,tag], fields))
 #           examples = [make_example(list(line), fields) for line in zip(src_f,tgt_f,tag_f)]
            super(DialogDataset, self).__init__(examples, fields, **kwargs)
        else:
            print('use rl')
            with codecs.open(src_path, encoding="utf8",errors='replace') as src_f, \
                codecs.open(tgt_path, encoding="utf8",errors='replace') as tgt_f:
                examples = []
                for src,tgt in zip(src_f,tgt_f):
                    if tgt.strip() == "_PAD":
                        tgt = PAD_WORD
                    examples.append(make_example([src,tgt], fields))
 #           examples = [make_example(list(line), fields) for line in zip(src_f,tgt_f,tag_f)]
            super(DialogDataset, self).__init__(examples, fields, **kwargs)
 



    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.src)


    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(DialogDataset, self).__reduce_ex__()


class OrderedIterator(torchtext.data.Iterator):
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
