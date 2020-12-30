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
    ids = [SOS_token] + ids
    return ids


def latentVar(latent_plain_sentence, vocab, oovs=[]):
    ids, oovs = latent2ids(latent_words=latent_plain_sentence, vocab=vocab, oovs=oovs)
    return ids, oovs


def outputVar(target_plain_sentence, vocab, oovs):
    ids = target2ids(target_words=target_plain_sentence, vocab=vocab, latent_oovs=oovs)
    ids = [SOS_token] + ids + [EOS_token]
    return ids


def setup_vocab(corpus, freq_threshold):

    print('building vocabulary ...')

    vocab = Vocab()
    vocab.wordCounter(corpus)
    vocab.addWord(freq_threshold=freq_threshold)

    # indicating word amount with respect to the setting threshold
    # print('\nword2count has content: \n{}\n'.format(vocab.word2count))
    # print('\nword2index has content: \n{}\n'.format(vocab.word2index))
    # print('\nindex2word has content: \n{}\n'.format(vocab.index2word))
    # print('\nthere are totally {} words\n'.format(vocab.n_words))

    return vocab


def setup_latent_sentence_vocab(filepath):
    with open(filepath, 'rb') as pkl_in:
        latent_sentence_vocab = pickle.load(pkl_in)

    latent_sentence_2_index = latent_sentence_vocab['latent_sentence_2_index']
    index_2_latent_sentence = latent_sentence_vocab['index_2_latent_sentence']

    return latent_sentence_2_index, index_2_latent_sentence


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

                latent_sentence_id = latent_sentence_2_index[latent_sentence]

                input_post_indices = inputVar(input_post, vocab)

                # in order to write oov_list into tsv file, we transform it into string of integers
                input_post_indices_lst = [str(index) for index in input_post_indices]
                input_post_indices_string = ' '.join(input_post_indices_lst)

                if input_post_indices_string not in out_collection:
                    out_collection[input_post_indices_string] = {'target_id': [], 'target_responses': []}
                out_collection[input_post_indices_string]['target_id'].append(str(latent_sentence_id))
                out_collection[input_post_indices_string]['target_responses'].append(target_response)

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

                if input_post_indices_string not in out_collection:
                    out_collection[input_post_indices_string] = []
                out_collection[input_post_indices_string].append(target_response)

            else:
                print("Dataset having wrong format.")


    # dictionary to list
    if train_or_valid:
        for key, values in out_collection.items():
            for target_response in values['target_responses']:
                out.append([key, ' '.join(values['target_id']), target_response])
    else:
        for key, values in out_collection.items():
            for value in values:
                out.append([key, value])

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
                target_response = dataPair[2]
                dataset.append([input_indices, target_indices, target_response])
            else:
                input_indices = [int(index) for index in dataPair[0].split()]
                target_response = dataPair[1]
                dataset.append([input_indices, target_response])

    return dataset


def batch(dataset, batch_size=4, train_or_valid=True):
    if train_or_valid:
        random.shuffle(dataset)

    batches = []
    batch_dictionary = {}
    for line in dataset:
        input_post_indices = line[0]
        if str(input_post_indices) not in batch_dictionary:
            batch_dictionary[str(input_post_indices)] = []
        batch_dictionary[str(input_post_indices)].append(line)

    for key, value in batch_dictionary.items():
        if len(value) > batch_size:
            pointer = 0
            while len(value) > batch_size:
                batches.append(value[pointer:pointer+batch_size])
                value = value[pointer+batch_size:]
            if len(value) != 0:
                batches.append(value)
        else:
            batches.append(value)

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
    input_mask = binaryMatrix(input_padList, value=PAD_token)
    input_mask = torch.ByteTensor(input_mask)
    input_mask = input_mask.transpose(0,1).contiguous()
    input_padVar = torch.LongTensor(input_padList)
    input_padVar = input_padVar.transpose(0,1).contiguous()    # needs to be transposed for batch first technique

    # latent sentence sample id
    tgt_ids = [pair[1] for pair in batch]

    # target response words
    tgt_responses = [pair[2] for pair in batch]

    return [input_ids, input_padVar, input_mask, input_lengths], [tgt_ids], [tgt_responses]


def batch2TestData(batch, train_or_valid=False):
    # for the sake of pytorch's operation efficiency, sort minibatch according to target ids length

    # input part
    input_ids = [pair[0] for pair in batch]
    input_lengths = torch.LongTensor([len(ids) for ids in input_ids])
    input_padList = zeroPadding(input_ids, fillvalue=PAD_token)
    input_mask = binaryMatrix(input_padList, value=PAD_token)
    input_mask = torch.ByteTensor(input_mask)
    input_mask = input_mask.transpose(0,1).contiguous()
    input_padVar = torch.LongTensor(input_padList)
    input_padVar = input_padVar.transpose(0,1).contiguous()    # needs to be transposed for batch first technique

    # target response words
    tgt_responses = [pair[1] for pair in batch]

    return [input_ids, input_padVar, input_mask, input_lengths], [tgt_responses]


def make_data(dataset, train_or_valid=True):
    # distribute dataset into batches
    batches = batch(dataset, batch_size=4, train_or_valid=train_or_valid)

    if train_or_valid:
        train_data = []
        for batch_ in batches:
            batch_inp_tuple, batch_sampler_tgt_tuple, batch_seq2seq_tgt_tuple = batch2TrainData(batch_, train_or_valid=train_or_valid)
            train_data.append([batch_inp_tuple, batch_sampler_tgt_tuple, batch_seq2seq_tgt_tuple])
        return train_data
    else:
        test_data = []
        for batch_ in batches:
            batch_inp_tuple, batch_seq2seq_tgt_tuple = batch2TestData(batch_, train_or_valid=train_or_valid)
            test_data.append([batch_inp_tuple, batch_seq2seq_tgt_tuple])
        return test_data


def make_2stage_data(selected_latent_sentences, target_responses, vocab):
    selected_latent_sentence_batch = []
    target_response_batch = []
    oovs_list = []
    for selected_latent_sentence, target_response in zip(selected_latent_sentences, target_responses):
        latent_sentence_indices, oovs = latentVar(selected_latent_sentence, vocab, oovs=[])
        target_response_indices = outputVar(target_response, vocab, oovs=oovs)

        selected_latent_sentence_batch.append(latent_sentence_indices)
        target_response_batch.append(target_response_indices)

        oovs_list.append(oovs)

    # prepare latent sentence info.
    latent_sentence_lengths = [len(ids) for ids in selected_latent_sentence_batch]
    latent_sentence_padList = zeroPadding(selected_latent_sentence_batch)
    latent_sentence_mask = binaryMatrix(latent_sentence_padList)
    latent_sentence_mask = torch.ByteTensor(latent_sentence_mask)
    latent_sentence_mask = latent_sentence_mask.transpose(0,1).contiguous()
    latent_sentence_padVar = torch.LongTensor(latent_sentence_padList)
    latent_sentence_padVar = latent_sentence_padVar.transpose(0,1).contiguous()

    # target response info.
    target_lengths = [len(ids)-1 for ids in target_response_batch]
    target_padList = zeroPadding(target_response_batch)
    target_mask = binaryMatrix(target_padList)
    target_mask = torch.ByteTensor(target_mask)
    target_padVar = torch.LongTensor(target_padList)
    dec_batch = target_padVar[:-1]
    target_batch = target_padVar[1:]
    dec_padding_mask = target_mask[1:]

    max_oov_len = max([len(oov) for oov in oovs_list])

    return [selected_latent_sentence_batch, latent_sentence_padVar, latent_sentence_mask, latent_sentence_lengths], [dec_batch, target_batch, dec_padding_mask, target_padVar, target_mask, target_lengths], [oovs_list, max_oov_len]
