from nltk.translate.bleu_score import corpus_bleu
import sys

weights = []
weights.append((1.0/1.0, ))
weights.append((1.0/2.0, 1.0/2.0, ))
weights.append((1.0/3.0, 1.0/3.0, 1.0/3.0, ))

out_file = sys.argv[1]
tgt_file = sys.argv[2]

o_sen = (open(out_file, 'r')).readlines()
t_sen = (open(tgt_file, 'r')).readlines()

outcorpus = []
i = 0
for s in o_sen:
    # if i % 4 == 0:
    outcorpus.append(s.split())
    i += 1
print(len(outcorpus))

tgtcorpus = []
tgt = []
for s in t_sen:
    tgt.append(s.split())
    i += 1
    if i % 4 == 0:
        for j in range(4):
            tgtcorpus.append(tgt)
        tgt = []

print('bleu-1: %s' %(corpus_bleu(tgtcorpus, outcorpus, weights[0])))
print('bleu-2: %s' %(corpus_bleu(tgtcorpus, outcorpus, weights[1])))
print('bleu-3: %s' %(corpus_bleu(tgtcorpus, outcorpus, weights[2])))
print('bleu-4: %s' %(corpus_bleu(tgtcorpus, outcorpus,(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, ))))


# format: arg1 -> prediction_file; arg2 -> target_file