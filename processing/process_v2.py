import sys
import csv


def write_():
    f = open(sys.argv[1], 'r')
    reader = csv.reader(f, delimiter='\t')

    fw = open('out.txt', 'w')
    i = 0
    for s in reader:
        fw.write(s[0][:-5])
        fw.write('\n')

def concat(pred_file, tgt_file, fw):
    sents = [l.strip('\n').strip() for l in pred_file.readlines()]
    reader = csv.reader(tgt_file, delimiter='\t')
    i = 0
    for s in reader:
        c = ' '.join([l[1:-1] for l in sents[i].split(' ')])
        fw.write(c)
        #fw.write(sents[i])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        fw.write(s[2])
        fw.write('\n')
        i += 1

def concatv2(pred_file, tgt_file, fw):	
    sents = [l.strip('\n').strip() for l in pred_file.readlines()]
    print(len(sents))
    reader = csv.reader(tgt_file, delimiter='\t')
    i = 0
    for s in reader:
        fw.write(s[0])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        c = ' '.join([l[1:-1] for l in sents[i].split(' ')])
        print(c)
        fw.write(c)
        fw.write('\n')
        i += 1
        print(i)

def concatv3(pred_file, tgt_file, pos_tgt_file, fw):
    pos_lines = pos_tgt_file.readlines()
    pos_lines = [[k[1:-1] for k in l.strip('\n').strip().split(' ')] for l in pos_lines]
    print(pos_lines[0])

    sents = [l.strip('\n').strip() for l in pred_file.readlines()]
    print(len(sents))
    reader = csv.reader(tgt_file, delimiter='\t')
    i = 0
    for s in reader:
        fw.write(s[0])
        fw.write('\t')
        fw.write(sents[i])
        fw.write('\t')
        fw.write(s[2])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        fw.write(' '.join(pos_lines[i]))
        fw.write('\n')
        i += 1



if __name__ == '__main__':
    #f = open(sys.argv[1], 'r')
    #tgt = open(sys.argv[2], 'r')
    #pos_tgt = open(sys.argv[3], 'r')
    #fw = open(sys.argv[4], 'w')

    #concatv3(f, tgt, pos_tgt, fw)
    write_()
