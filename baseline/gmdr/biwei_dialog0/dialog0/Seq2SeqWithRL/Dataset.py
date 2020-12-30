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
import pdb

class RLDataset(torchtext.data.Dataset):


    def __init__(self, data_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist
        with codecs.open(data_path, encoding="utf8",errors='ignore') as train_f:
            examples = []
            for line in train_f:
                data = line.strip().split('\t')
                # pdb.set_trace()
                if len(data) != 3:
                    print("miss: %s"%(line.strip()))
                    continue
                else:
                    # pdb.set_trace()
                    src,tag,tgt = data[0],data[1],data[2]
                examples.append(make_example([src,tgt,tag], fields))
#           examples = [make_example(list(line), fields) for line in zip(src_f,tgt_f,tag_f)]
        super(RLDataset, self).__init__(examples, fields, **kwargs)




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
        return super(RLDataset, self).__reduce_ex__()

class InferDataset(torchtext.data.Dataset):


    def __init__(self, data_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist
        with codecs.open(data_path, encoding="utf8",errors='ignore') as test_f:
            examples = []
            for line in test_f:
                src = line.split('\t')[0]
                # src,tag = data[0],data[1]

                examples.append(make_example([src], fields))
#           examples = [make_example(list(line), fields) for line in zip(src_f,tgt_f,tag_f)]
        super(InferDataset, self).__init__(examples, fields, **kwargs)



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
        return super(InferDataset, self).__reduce_ex__()
