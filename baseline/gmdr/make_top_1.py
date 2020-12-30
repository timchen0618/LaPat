import pdb
import csv

input_file = 'result_rl_test.test'
output_file = 'result_rl_test_top1.test'

with open(input_file, 'r') as r_file:
    reader = csv.reader(r_file, delimiter='\t')

    with open(output_file, 'w') as w_file:
        writer = csv.writer(w_file, delimiter='\t')

        for idx, lines in enumerate(reader):
            score = float(lines[1].lstrip('tensor(')[:7])
            if idx % 3 == 0:
                scores = []
                sentences = []

            scores.append(score)
            sentences.append(lines)

            if idx % 3 == 2:
                max_index = scores.index(max(scores))
                max_sentence = sentences[max_index]
                
                writer.writerow(max_sentence)
