import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, shared_embeddings, model, vocab, device):
        super(Sampler, self).__init__()

        self.shared_embeddings = shared_embeddings
        self.model = model
        self.vocab = vocab

        self.device = device

    def forward(self, enc_batch, enc_lens):
        # Run Encoder
        input_embed = self.shared_embeddings(enc_batch)
        post_encoder_outputs, post_encoder_hidden = self.model.encoder(input_embed, enc_lens)
        # post_encoder_hidden: (layers x batch x directions*hidden_dim)
        #                      (1 x batch_size x 2*500)

        # Run Classifier
        log_probs = self.model.classifier(post_encoder_hidden)
        selected_latent_sentence_id = log_probs.argmax(dim=-1)

        return log_probs, selected_latent_sentence_id


    def compute_loss(self, log_probs, tgt_ids):
        step_losses = []
        for index, tgt_id in enumerate(tgt_ids):
            true_dist = log_probs[index].data.clone()
            true_dist = true_dist.fill_(0.).unsqueeze(0)
            true_dist.scatter_(1, torch.LongTensor(tgt_id).to(self.device).unsqueeze(0), 1.)
            loss = -(true_dist*log_probs[index]).sum()
            step_losses.append(loss.unsqueeze(0))

        avg_loss = torch.cat(step_losses, dim=0).mean()

        return avg_loss


    def drop_checkpoint(self, epoch, fname):
        torch.save({'sampler_shared_embeddings_state_dict': self.shared_embeddings.state_dict(),
                    'sampler_encoder_state_dict': self.model.encoder.state_dict(),
                    'sampler_classifier_state_dict': self.model.classifier.state_dict(),
                    'epoch': epoch},
                    fname)


    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.shared_embeddings.load_state_dict(cpnt['sampler_shared_embeddings_state_dict'])
        self.model.encoder.load_state_dict(cpnt['sampler_encoder_state_dict'])
        self.model.classifier.load_state_dict(cpnt['sampler_classifier_state_dict'])
        epoch = cpnt['epoch']
        return epoch
