import sys
import csv



def concat_test_pos(f, f_pred, fw):
    reader = csv.reader(f, delimiter='\t')
    pred = f_pred.readlines()

    i = 0
    for s in reader:
        
        p = pred[i].strip('\n').strip()
        fw.write(s[0])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        fw.write('...')
        fw.write('\t')
        fw.write(p)
        fw.write('\n')


def concat_post_pred(f, f_pred, fw):
    reader = csv.reader(f, delimiter='\t')
    pred = f_pred.readlines()

    i = 0
    for s in reader:
        
        p = pred[i].strip('\n').strip()
        fw.write(s[0])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        fw.write(s[2])
        fw.write('\t')
        fw.write(p)
        fw.write('\n')

        # str_ = ""
        # str_ += s[0].ljust(40)
        # str_ += s[1].ljust(30)
        # str_ += p.ljust(30)
        # fw.write(str_)
        fw.write('\n')
        i+=1
        print(i)

        # i+=1

def concat_post_10pos(f, fw, l):
    reader = csv.reader(f, delimiter='\t')
    for s in reader:
        for pos in l:
            fw.write(s[0])
            fw.write('\t')
            fw.write('.')
            fw.write('\t')
            fw.write('.')
            fw.write('\t')
            fw.write(pos)
            fw.write('\n')

def extract_ref_sent(f, fw):
    reader = csv.reader(f, delimiter='\t')
    for s in reader:
        fw.write(s[1])
        fw.write('\n')


if __name__ == '__main__':
    # concat post pred
    f = open(sys.argv[1], 'r')
    f_pred = open(sys.argv[2], 'r')
    fw = open(sys.argv[3], 'w')
    concat_post_pred(f, f_pred, fw)

    # # concat_test_pos
    # f = open(sys.argv[1], 'r')
    # f_pred = open(sys.argv[2], 'r')
    # fw = open(sys.argv[3], 'w')
    # concat_test_pos(f, f_pred, fw)

    # f = open(sys.argv[1], 'r')
    # fw = open(sys.argv[2], 'w')
    # l = ['n w n w', 'v n w v n w', 'n', 'n w n w n w', 'v n n n w', 'n n w n n w', 'n n n w', 'i w i w', 'n n n n w', 'n n n n n w']
    # concat_post_10pos(f, fw, l)

    # f = open(sys.argv[1], 'r')
    # fw = open(sys.argv[2], 'w')
    # extract_ref_sent(f, fw)