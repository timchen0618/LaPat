import csv
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


def Plain_Text_to_Train_Data(corpus_path, out_file, vocab):

    print('saving training data ...')

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
            elif len(line) == 5:
                input_post = line[0]
                latent_sentence = None
                target_response = line[1]
                complete_postags = line[2]
                processed_postags = line[3]
                pos_id = line[4]
            elif len(line) == 3:
                input_post = line[0]
                latent_sentence = line[2]
                target_response = line[1]
            else:
                print("Dataset having wrong format.")

            input_indices = inputVar(input_post, vocab)
            latent_indices, oov_list = latentVar(latent_sentence, vocab, oovs=[])
            target_indices = outputVar(target_response, vocab, oovs=oov_list)

            # in order to write oov_list into tsv file, we transform it into string of integers
            input_indices_lst = [str(index) for index in input_indices]
            latent_indices_lst = [str(index) for index in latent_indices]
            target_indices_lst = [str(index) for index in target_indices]
            oov_list = [str(index) for index in oov_list]
            input_string = ' '.join(input_indices_lst)
            latent_string = ' '.join(latent_indices_lst)
            target_string = ' '.join(target_indices_lst)
            oov_string = ' '.join(oov_list)

            out.append([input_string, latent_string, target_string, oov_string])


    with open(out_file, 'w') as tsv_out:
        tsv_writer = csv.writer(tsv_out, delimiter='\t')

        for outLine in out:
            tsv_writer.writerow(outLine)


############### Saving input_word_indices, target_word_indices & oov_list for each pair #########################
#################################################################################################################


####################################################################################################
############################## Load saved dataset & set up mini-batch ##############################


def load_format(out_file):
    dataset = []
    with open(out_file, 'r') as tsv_in:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')

        for dataPair in tsv_reader:
            input_indices = [int(index) for index in dataPair[0].split()]
            latent_indices = [int(index) for index in dataPair[1].split()]
            target_indices = [int(index) for index in dataPair[2].split()]
            oov_list = [oov_word for oov_word in dataPair[3].split()]

            dataset.append((input_indices, latent_indices, target_indices, oov_list))

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



def batch2TrainData(batch, vocab_size, train_or_valid=True):
    # for the sake of pytorch's operation efficiency, sort minibatch according to target ids length
    if train_or_valid:
        batch.sort(key=lambda x: len(x[2]), reverse=True)

    if train_or_valid:
        # input part
        input_ids = [pair[0] for pair in batch]
        input_lengths = [len(ids) for ids in input_ids]
        input_padList = zeroPadding(input_ids, fillvalue=PAD_token)
        input_mask = binaryMatrix(input_padList)
        input_mask = torch.ByteTensor(input_mask)    # needs to be transposed for batch first technique
        input_padVar = torch.LongTensor(input_padList)    # needs to be transposed for batch first technique
        input_batch_extend_vocab = input_padVar
        input_oov_mask = input_batch_extend_vocab < vocab_size
        input_padVar = input_oov_mask.long() * input_batch_extend_vocab

        # latent sentence part
        latent_ids = [pair[1] for pair in batch]
        latent_lengths = [len(ids) for ids in latent_ids]
        latent_padList = zeroPadding(latent_ids, fillvalue=PAD_token)
        latent_mask = binaryMatrix(latent_padList)
        latent_mask = torch.ByteTensor(latent_mask)
        latent_padVar = torch.LongTensor(latent_padList)
        latent_batch_extend_vocab = latent_padVar
        latent_oov_mask = latent_batch_extend_vocab < vocab_size
        latent_padVar = latent_oov_mask.long() * latent_batch_extend_vocab

        # target part
        target_ids = [pair[2] for pair in batch]
        target_lengths = [len(ids)-1 for ids in target_ids]
        max_target_length = max(target_lengths)
        target_padList = zeroPadding(target_ids, fillvalue=PAD_token)
        target_mask = binaryMatrix(target_padList)
        target_mask = torch.ByteTensor(target_mask)
        target_padVar = torch.LongTensor(target_padList)
        dec_batch = target_padVar[:-1]
        dec_batch_extend_vocab = dec_batch
        dec_batch_oov_mask = dec_batch_extend_vocab < vocab_size
        dec_batch = dec_batch_oov_mask.long() * dec_batch_extend_vocab
        target_batch = target_padVar[1:]
        dec_padding_mask = target_mask[1:]

        oov_list = [pair[3] for pair in batch]
        max_oov_len = max([len(oov) for oov in oov_list])

        return (input_padVar, input_mask, input_lengths, input_batch_extend_vocab), (latent_padVar, latent_mask, latent_lengths, latent_batch_extend_vocab), (dec_batch, target_batch, dec_padding_mask, target_padVar, target_mask, target_lengths, max_target_length), (oov_list, max_oov_len)
    else:
        input_ids = [pair[0] for pair in batch]
        input_lengths = [len(ids) for ids in input_ids]
        input_padList = zeroPadding(input_ids, fillvalue=PAD_token)
        input_mask = binaryMatrix(input_padList)
        input_mask = torch.ByteTensor(input_mask)    # needs to be transposed for batch first technique
        input_padVar = torch.LongTensor(input_padList)    # needs to be transposed for batch first technique
        input_batch_extend_vocab = input_padVar
        input_oov_mask = input_batch_extend_vocab < vocab_size
        input_padVar = input_oov_mask.long() * input_batch_extend_vocab

        # latent sentence part
        latent_ids = [pair[1] for pair in batch]
        latent_lengths = [len(ids) for ids in latent_ids]
        latent_padList = zeroPadding(latent_ids, fillvalue=PAD_token)
        latent_mask = binaryMatrix(latent_padList)
        latent_mask = torch.ByteTensor(latent_mask)
        latent_padVar = torch.LongTensor(latent_padList)
        latent_batch_extend_vocab = latent_padVar
        latent_oov_mask = latent_batch_extend_vocab < vocab_size
        latent_padVar = latent_oov_mask.long() * latent_batch_extend_vocab

        # target part
        target_ids = [pair[2] for pair in batch]
        target_lengths = [len(ids)-1 for ids in target_ids]
        max_target_length = max(target_lengths)
        target_padList = zeroPadding(target_ids, fillvalue=PAD_token)
        target_mask = binaryMatrix(target_padList)
        target_mask = torch.ByteTensor(target_mask)
        target_padVar = torch.LongTensor(target_padList)
        dec_batch = target_padVar[:-1]
        dec_batch_extend_vocab = dec_batch
        dec_batch_oov_mask = dec_batch_extend_vocab < vocab_size
        dec_batch = dec_batch_oov_mask.long() * dec_batch_extend_vocab
        target_batch = target_padVar[1:]
        dec_padding_mask = target_mask[1:]

        oov_list = [pair[2] for pair in batch]
        max_oov_len = max([len(oov) for oov in oov_list])

        return (input_padVar, input_mask, input_lengths, input_batch_extend_vocab), (latent_padVar, latent_mask, latent_lengths, latent_batch_extend_vocab), (target_padVar), (oov_list, max_oov_len)


def make_data(dataset, vocab_size, train_or_valid=True):
    # distribute dataset into batches
    batches = batch(dataset, batch_size=64, train_or_valid=train_or_valid)

    train_data = []
    for batch_ in batches:
        batch_inp_tuple, batch_latent_tuple, batch_tgt_tuple, oov_tuple = batch2TrainData(batch_, vocab_size, train_or_valid=train_or_valid)
        train_data.append((batch_inp_tuple, batch_latent_tuple, batch_tgt_tuple, oov_tuple))
    return train_data
