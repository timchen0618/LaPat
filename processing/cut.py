import os
import sys
import csv

f1 = open(sys.argv[1], 'r')
fw = open(sys.argv[2], 'w')
count = int(sys.argv[3])

reader = csv.reader(f1, delimiter='\t')
#sent = f1.readlines()
i = 0
prev = ""
for s in reader:
    if s[0] != prev:
        fw.write(s[0])
        fw.write('\t')
        fw.write(s[1])
        fw.write('\t')
        fw.write(s[2])
        fw.write('\t')
        fw.write(s[3])
        fw.write('\n')
        prev = s[0]
        i += 1
    if i == count:
        break
f1.close()
fw.close()