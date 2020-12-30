"""
DIR=
python3 ${DIR}/infer_rl.py \
    -src_in \
    -tgt_out \
    -model \
    -data \
    -config ./config.yml \
    -gpuid \
    -beam_size 3 \
    -topk_tag 1 \
    -decode_max_length 20
"""
import os
import argparse
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument("-test_data", type=str)
parser.add_argument("-test_out", type=str)
parser.add_argument("-vocab", type=str)
parser.add_argument("-model", type=str)
parser.add_argument("-config", type=str)
parser.add_argument("-dump_beam", default="", type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)
parser.add_argument("-beam_size", type=int)
parser.add_argument("-topk_tag", type=int)
parser.add_argument("-decode_max_length", type=int)
parser.add_argument("-num_worker", type=int)
args = parser.parse_args()


os.system("mkdir -p ./multi_process_tmp")
os.system("cp %s ./multi_process_tmp"%(args.test_data))
num_line = 0
test_data = os.path.join("./multi_process_tmp",os.path.basename(args.test_data))
with open(test_data,'r') as f:
    lines = f.readlines()
    num_line = len(lines)
print(num_line)
lines_per_segment = num_line//args.num_worker

prefix_str = "%s.tmp_"%(os.path.basename(args.test_data))

os.system("split -l %d %s -d -a 1 %s"% \
            (lines_per_segment, test_data, os.path.join("./multi_process_tmp",prefix_str)))


file_list = os.listdir('./multi_process_tmp')

candidate_file = []
print(file_list)
for f_str in  file_list:
    if prefix_str in f_str:
        candidate_file.append(f_str)
status = [0 for _ in range(args.num_worker)]
for i,cand_f in enumerate(candidate_file):
    print("start worker %d"%(i))
    status[i] = subprocess.call('python3 ./biwei-dialog0/infer_rl.py \
                    -test_data ./multi_process_tmp/%s \
                    -test_out ./multi_process_tmp/%s_res \
                    -model %s \
                    -vocab %s \
                    -config %s \
                    -gpuid %s \
                    -beam_size %s \
                    -topk_tag %s \
                    -decode_max_length %s'%(
                    cand_f,
                    cand_f,
                    args.model,
                    args.vocab,
                    args.config,
                    args.gpuid[0],
                    args.beam_size,
                    args.topk_tag,
                    args.decode_max_length
                    ),shell=True)

while status.all():
    pass

print(status.all())