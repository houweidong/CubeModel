from torch import nn
from model.cubenet import CubeNet


def generate_model(opt, tasks=None):
    assert opt.model in ['cube']
    if opt.model == 'cube':
        # TODO Just assume task == attribute right now. Will need to update
        assert tasks is not None
        model = CubeNet(opt.conv, tasks, pretrained=opt.pretrain, img_size=opt.person_size,
                        attention=opt.attention, dropout=opt.dropout, at=opt.at, at_loss=opt.at_loss)
    else:
        raise Exception('Unsupported model {}'.format(opt.conv))
    mean, std = model.mean, model.std

    model = model.cuda()
    parameters = model.parameters()
    # parameters = model.get_parameter_groups()

    return model, parameters, mean, std
