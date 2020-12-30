import csv
import os
import sys

def main(f, fw):
    LABEL_LIMIT=12
    prev = ""
    labels = []
    reader = csv.reader(f, delimiter='\t')

    for s in reader:
        # print(labels, '  ', s[0])
        if s[0] != prev:
            if len(labels) != 0:
                while(len(labels) > LABEL_LIMIT):
                    l_to_write, labels = labels[:LABEL_LIMIT], labels[LABEL_LIMIT:]
                    write_labels(fw, prev, l_to_write)
                write_labels(fw, prev, labels)
            prev = s[0] 
            labels = []
        labels.append(s[5])

def write_labels(fw, prev, labels):
    fw.write(prev)
    fw.write('\t')
    label_str = ""
    for l in labels:
        label_str += l
        label_str += ' '
    fw.write(label_str[:-1])
    fw.write('\n')



if __name__ == '__main__':
    filename = sys.argv[1]
    f = open(filename, 'r')
    fw = open(filename[:-4].replace('latent', 'sampler') + '_multi.tsv', 'w')
    main(f, fw)