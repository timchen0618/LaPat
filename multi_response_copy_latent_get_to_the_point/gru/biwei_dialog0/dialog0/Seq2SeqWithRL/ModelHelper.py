from dialog0.Seq2SeqWithRL.Model import Seq2SeqWithRL


def create_rl_model(sampler_model, seq2seq_model):
    seq2seq_with_rl_model = Seq2SeqWithRL(sampler_model=sampler_model,
                                          seq2seq_model=seq2seq_model)

    return seq2seq_with_rl_model
