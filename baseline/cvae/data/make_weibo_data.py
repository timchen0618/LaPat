import pickle
import csv
import pdb
from copy import deepcopy
weibo_file_train = '/share/home/alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_train_meteor.tsv'
weibo_file_valid = '/share/home/alexchao2007/code/weibo_data_final/10000/pos_unprocessed_10000_dev_meteor.tsv'
weibo_file_test = '/share/home/alexchao2007/code/weibo_data_final/10000/new_data_test_10000_meteor.tsv'
cvae_file = 'full_swda_clean_42da_sentiment_dialog_corpus.p'
out_file = 'weibo_transformed_cvae_splitted.p'
# out_file = 'test.p'

cvae_data = pickle.load(open(cvae_file, 'rb'))

train_transform = []

with open(weibo_file_train, 'r') as read_file:
    train_reader = csv.reader(read_file, delimiter='\t')
    for idx, lines in enumerate(train_reader):
            if idx % 1000 == 0:
                if idx == 0:
                    transform_batch = []
                    post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                    response = ('B', lines[2], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                    transform_batch.append(post)
                    transform_batch.append(response)
                    continue
                
                transform_batch_temp = deepcopy(cvae_data['test'][0])
                transform_batch_temp['utts'] = transform_batch
                train_transform.append(transform_batch_temp)
                transform_batch = []
                print('----Finish {}----'.format(idx))

            post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
            response = ('B', lines[2], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

            transform_batch.append(post)
            transform_batch.append(response)

cvae_data['train'] = train_transform

dev_transform = []

with open(weibo_file_valid, 'r') as read_file:
    train_reader = csv.reader(read_file, delimiter='\t')
    for idx, lines in enumerate(train_reader):
            if idx % 200 == 0:
                if idx == 0:
                    transform_batch = []
                    post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                    response = ('B', lines[2], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                    transform_batch.append(post)
                    transform_batch.append(response)
                    continue
                
                transform_batch_temp = deepcopy(cvae_data['test'][0])
                transform_batch_temp['utts'] = transform_batch
                dev_transform.append(transform_batch_temp)
                transform_batch = []
                print('----Finish {}----'.format(idx))

            post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
            response = ('B', lines[2], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

            transform_batch.append(post)
            transform_batch.append(response)

cvae_data['valid'] = dev_transform


test_transform = []

with open(weibo_file_test, 'r') as read_file:
    train_reader = csv.reader(read_file, delimiter='\t')
    for idx, lines in enumerate(train_reader):
            if idx % 200 == 0 or idx == 3199:
                if idx == 0:
                    transform_batch = []
                    post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                    response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                    transform_batch.append(post)
                    transform_batch.append(response)
                    continue
                
                transform_batch_temp = deepcopy(cvae_data['test'][0])
                transform_batch_temp['utts'] = transform_batch
                test_transform.append(transform_batch_temp)
                transform_batch = []

                post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                transform_batch.append(post)
                transform_batch.append(response)

                print('----Finish {}----'.format(idx))

            post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
            response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

            transform_batch.append(post)
            transform_batch.append(response)

cvae_data['test'] = test_transform


test_transform = []

with open(weibo_file_test, 'r') as read_file:
    train_reader = csv.reader(read_file, delimiter='\t')
    for idx, lines in enumerate(train_reader):
            if idx % 200 == 0:
                if idx == 0:
                    transform_batch = []
                    post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                    response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                    transform_batch.append(post)
                    transform_batch.append(response)
                    continue
                
                transform_batch_temp = deepcopy(cvae_data['test'][0])
                transform_batch_temp['utts'] = transform_batch
                test_transform.append(transform_batch_temp)
                transform_batch = []

                post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                transform_batch.append(post)
                transform_batch.append(response)

                print('----Finish {}----'.format(idx))
                continue

            if idx == 3199:
                post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
                response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

                transform_batch.append(post)
                transform_batch.append(response)
                transform_batch_temp = deepcopy(cvae_data['test'][0])
                transform_batch_temp['utts'] = transform_batch
                test_transform.append(transform_batch_temp)


            post = ('A', lines[0], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])
            response = ('B', lines[1], ['statement-non-opinion', [0.149, 0.851, 0.0, -0.4215]])

            transform_batch.append(post)
            transform_batch.append(response)

cvae_data['test_real'] = test_transform

with open(out_file, 'wb') as w_file:
    pickle.dump(cvae_data, w_file, protocol=2)

