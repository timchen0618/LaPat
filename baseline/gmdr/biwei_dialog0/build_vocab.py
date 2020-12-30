import dialog0.Seq2Seq.IO as IO
import argparse
import torch
import dialog0.Utils as utils
from dialog0.Seq2SeqWithRL.Dataset import RLDataset
from dialog0.Seq2Seq.Dataset import SeqDataset
parser = argparse.ArgumentParser()
parser.add_argument('-train_data', type=str)
parser.add_argument('-save_data', type=str)
parser.add_argument('-config', type=str)
args = parser.parse_args()

config = utils.load_config(args.config)

if config['Misc']['random_seed'] > 0:
    torch.manual_seed(config['Misc']['random_seed'])

fields = IO.get_fields()
print("Building Training...")
train = SeqDataset(
    data_path=args.train_data,
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"]),
            ('tag', fields["tag"])])
print("Building Vocab...")
IO.build_vocab(train, config)

print("Saving fields")
torch.save(IO.save_vocab(fields),open(args.save_data+'.vocab.pkl', 'wb'))
# train.fields = []
# torch.save(train, open(args.save_data+'.train.pkl', 'wb'))
