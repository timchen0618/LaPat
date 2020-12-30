from model_HGFU import HGFU
import torch.nn as nn

models = ['HGFU', 'Seq2Seq']

def build_model(model_name, config, use_cuda, data_utils):
    assert model_name in models
    # module = __import__(module_name)
    # class_ = getattr(module, class_name)
    # instance = class_()

    import importlib
    module = importlib.import_module('model_'+model_name)
    class_ = getattr(module, model_name)
    config = config[model_name]
    config['vocab_size'] = data_utils.vocab_size
    config['padding_idx'] = data_utils.pad

    model = class_(config, use_cuda)
    if use_cuda:
        model = model.cuda()

    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.xavier_uniform_(model)
            nn.init.uniform_(p, a=-0.1, b=0.1)
    return model

