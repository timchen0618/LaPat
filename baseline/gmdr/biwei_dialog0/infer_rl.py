import torch
import pdb
import argparse
import codecs
import dialog0.Seq2SeqWithRL.IO as IO
from dialog0.Seq2SeqWithRL.Infer import Infer
import json
from torch import cuda
from dialog0.Seq2Seq.ModelHelper import create_seq2seq_tag_model
from dialog0.TagSampler.ModelHelper import create_tag_sampler
from dialog0.Seq2SeqWithRL.ModelHelper import create_seq2seq_rl_model
import dialog0.Utils as utils
from dialog0.Seq2SeqWithRL.Dataset import InferDataset
import progressbar
def indices_lookup(indices,fields):

    words = [fields['tgt'].vocab.itos[i] for i in indices]
    sent = ' '.join(words)

    return sent




def inference_file(infer, data_iter, tgt_fout, fields, topk_tag, num_cluster, use_cuda, writer=None):


    print('start translating ...')
    with codecs.open(tgt_fout, 'wb', encoding='utf8') as tgt_file:
        # bar = progressbar.ProgressBar()
        for idx,batch in enumerate(data_iter):
            clusters = infer.sample_tag_with_kmeans(batch,topk_tag,num_cluster)
            clusters = sorted(clusters,key=lambda x:len(x))
            # pdb.set_trace() 
            for idx,cluster in enumerate(clusters):
                if len(cluster) == 0 :
                    tgt_file.write("None"+'\n')
                    continue
                if idx == 3:
                    break

                clusters[idx] = sorted(cluster,key=lambda x:-(x[1]))
                cluster = sorted(cluster,key=lambda x:-(x[1]))

                ret = infer.inference_batch(batch,cluster[0][0],cluster[0][1])
                src_text = indices_lookup(batch.src[0][:,0].data.tolist(), fields)
                tgt_text = indices_lookup(ret['predictions'][0][0], fields)
                ctx_attn = ret["ctx_attns"][0][0].cpu().numpy().tolist()
                tag_attn = ret["tag_attns"][0][0][0].cpu().numpy().tolist()
                selected_tag = ret['tag']
                selected_tag_score=ret['tag_score']
                score = ret['scores'][0][0]
                tag_c = ret['tag_c'].cpu().numpy().tolist()
                
                # clusters[idx] = sorted(cluster,key=lambda x:-(x[1]))

                top5_tag_indices = [c[0] for c in clusters[idx][:5]]
                top5_tag = indices_lookup(top5_tag_indices,fields)
                tgt_file.write(tgt_text+'\t'+str(score)+'\t'+selected_tag+'\t'+str(selected_tag_score)+'\t'+top5_tag+'\n')


                save_data = {
                    "tag":selected_tag,
                    "ctx_attn":ctx_attn,
                    "tag_attn":tag_attn,
                    "src_text":src_text,
                    "tgt_text":tgt_text,
                    "tag_score":selected_tag_score,
                    "score":score,
                    "tag_c":tag_c
                }
                if writer:
                    abc=123
                    #pdb.set_trace()
                    # writer.write(json.dumps(save_data)+'\n')


            if idx % 100 == -1 % 100:
                print("processed %d"%(idx))
                    

        # if dump_beam:
        #     print('dump beam ....')
        #     beam_trace = infer.beam_accum
        #     with codecs.open(dump_beam,'w',encoding='utf8') as f:
        #         json.dump(beam_trace,f)



def main():

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
    parser.add_argument("-num_cluster", type=int)
    parser.add_argument("-decode_max_length", type=int)
    args = parser.parse_args()
    config = utils.load_config(args.config)

    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = IO.load_fields(
                torch.load(args.vocab))

    infer_dataset = InferDataset(
        data_path=args.test_data,
        fields=[('src', fields["src"])])

    data_iter = IO.InferIterator(
                dataset=infer_dataset, device=args.gpuid[0],
                batch_size=1, train=False, sort=False,
                sort_within_batch=True, shuffle=False)
    # Build model.

    seq2seq_model = create_seq2seq_tag_model(config, fields)
    sampler_model = create_tag_sampler(config, fields)
    model = create_seq2seq_rl_model(config, fields, sampler_model, seq2seq_model)


    print('Loading parameters ...')
    if args.model:
        model.load_checkpoint(args.model)
    if use_cuda:
        model = model.cuda()

    infer = Infer(model=model, 
                    fields=fields,
                    beam_size=args.beam_size, 
                    n_best=1,
                    max_length=args.decode_max_length,
                    global_scorer=None,
                    cuda=use_cuda,
                    beam_trace=True if args.dump_beam else False)


    writer = None
    if args.dump_beam:
        writer = open(args.dump_beam,'w',encoding='utf8')


    inference_file(infer, data_iter, args.test_out, fields, args.topk_tag, args.num_cluster, use_cuda, writer)
    

if __name__ == '__main__':
    main()

