import csv
import pickle
import random
import torch
import itertools
from dialog0.Seq2Seq.Vocab_conference import Vocab, input2ids, latent2ids, target2ids


#################################################################################################################
############### Saving input_word_indices, target_word_indices & oov_list for each pair #########################

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3



def inputVar(input_plain_sentence, vocab):
    ids = input2ids(input_words=input_plain_sentence, vocab=vocab)
    return ids


def latentVar(latent_plain_sentence, vocab, oovs=[]):
    ids, oovs = latent2ids(latent_words=latent_plain_sentence, vocab=vocab, oovs=oovs)
    return ids, oovs


def outputVar(target_plain_sentence, vocab, oovs):
    ids = target2ids(target_words=target_plain_sentence, vocab=vocab, latent_oovs=oovs)
    ids = [SOS_token] + ids + [EOS_token]
    return ids


def setup_latent_sentence_vocab(filepath):
    with open(filepath, 'rb') as pkl_in:
        latent_sentence_vocab = pickle.load(pkl_in)

    latent_sentence_2_index = latent_sentence_vocab['latent_sentence_2_index']
    index_2_latent_sentence = latent_sentence_vocab['index_2_latent_sentence']

    return latent_sentence_2_index, index_2_latent_sentence


def setup_vocab(corpus, freq_threshold):

    print('building vocabulary ...')

    vocab = Vocab()
    vocab.wordCounter(corpus)
    vocab.addWord(freq_threshold=freq_threshold)

    return vocab


def Plain_Text_to_Train_Data(corpus_path, out_file, vocab, latent_sentence_2_index, train_or_valid=True):

    print('saving training data ...')

    out_collection = {}
    out = []
    with open(corpus_path, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for line in tsv_reader:
            if len(line) == 6:
                input_post = line[0]
                latent_sentence = line[1]
                target_response = line[2]
                complete_postags = line[3]
                processed_postags = line[4]
                pos_id = line[5]

                input_post_indices = inputVar(input_post, vocab)
                input_post_indices_lst = [str(index) for index in input_post_indices]
                input_post_indices_string = ' '.join(input_post_indices_lst)

                latent_sentence_id = latent_sentence_2_index[latent_sentence]

                if input_post_indices_string not in out_collection:
                    out_collection[input_post_indices_string] = []
                out_collection[input_post_indices_string].append(str(latent_sentence_id))
            elif len(line) == 5:
                input_post = line[0]
                latent_sentence = None
                target_response = line[1]
                complete_postags = line[2]
                processed_postags = line[3]
                pos_id = line[4]

                input_post_indices = inputVar(input_post, vocab)
                input_post_indices_lst = [str(index) for index in input_post_indices]
                input_post_indices_string = ' '.join(input_post_indices_lst)

                out.append([input_post_indices_string])
            else:
                print("Dataset having wrong format.")


    # dictionary to list
    if train_or_valid:
        for key, value in out_collection.items():
            out.append([key, ' '.join(value)])

    with open(out_file, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for outLine in out:
            tsv_writer.writerow(outLine)


############### Saving input_word_indices, target_word_indices & oov_list for each pair #########################
#################################################################################################################


####################################################################################################
############################## Load saved dataset & set up mini-batch ##############################


def load_format(out_file, train_or_valid=True):
    dataset = []
    with open(out_file, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for dataPair in tsv_reader:
            if train_or_valid:
                input_indices = [int(index) for index in dataPair[0].split()]
                target_indices = [int(index) for index in dataPair[1].split()]
                dataset.append((input_indices, target_indices))
            else:
                input_indices = [int(index) for index in dataPair[0].split()]
                dataset.append((input_indices))

    return dataset


def batch(dataset, batch_size=128, train_or_valid=True):
    if train_or_valid:
        random.shuffle(dataset)

    batches = []
    pointer = 0
    while pointer < len(dataset):
        if pointer + batch_size <= len(dataset):
            batches.append(dataset[pointer:pointer+batch_size])
            pointer = pointer + batch_size
        else:
            batches.append(dataset[pointer:])
            pointer = len(dataset)

    return batches


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def batch2TrainData(batch, train_or_valid=True):
    # for the sake of pytorch's operation efficiency, sort minibatch according to target ids length
    if train_or_valid:
        batch.sort(key=lambda x: len(x[1]), reverse=True)

    # input part
    input_ids = [pair[0] for pair in batch]
    input_lengths = torch.LongTensor([len(ids) for ids in input_ids])
    input_padList = zeroPadding(input_ids, fillvalue=PAD_token)
    input_padVar = torch.LongTensor(input_padList).transpose(0,1)    # needs to be transposed for batch first technique

    # latent sentence sample id
    tgt_ids = [pair[1] for pair in batch]

    return (input_ids, input_padVar, input_lengths), (tgt_ids)


def batch2TestData(batch, train_or_valid=False):
    # for the sake of pytorch's operation efficiency, sort minibatch according to target ids length

    # input part
    input_ids = [pair for pair in batch]
    input_lengths = torch.LongTensor([len(ids) for ids in input_ids])
    input_padList = zeroPadding(input_ids, fillvalue=PAD_token)
    input_padVar = torch.LongTensor(input_padList).transpose(0,1)    # needs to be transposed for batch first technique

    return (input_ids, input_padVar, input_lengths)


def make_data(dataset, train_or_valid=True):
    # distribute dataset into batches
    batches = batch(dataset, batch_size=128, train_or_valid=train_or_valid)

    if train_or_valid:
        train_data = []
        for batch_ in batches:
            batch_inp_tuple, batch_tgt_tuple = batch2TrainData(batch_, train_or_valid=train_or_valid)
            train_data.append((batch_inp_tuple, batch_tgt_tuple))
        return train_data
    else:
        test_data = []
        for batch_ in batches:
            batch_inp_tuple = batch2TestData(batch_, train_or_valid=train_or_valid)
            test_data.append(batch_inp_tuple)
        return test_data
