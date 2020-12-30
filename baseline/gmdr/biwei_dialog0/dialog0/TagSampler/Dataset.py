from __future__ import unicode_literals
from __future__ import print_function
import torchtext
from collections import defaultdict,Counter
import codecs
from itertools import count

class SamplerDataset(torchtext.data.Dataset):


    def __init__(self, data_path, fields, **kwargs):

        make_example = torchtext.data.Example.fromlist

        with codecs.open(data_path, encoding="utf8",errors='ignore') as train_f:
            examples = []
            for line in train_f:
                data = line.strip().split('\t')
                # if len(data) != 2:
                #     continue
                # else:                
                src,tag = data[0],data[1]
                examples.append(make_example([src,tag], fields))
        super(SamplerDataset, self).__init__(examples, fields, **kwargs)



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
        return super(SamplerDataset, self).__reduce_ex__()