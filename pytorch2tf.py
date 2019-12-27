import keras
import numpy as np
from torch.autograd import Variable
import torch
from pytorch2keras.converter import pytorch_to_keras
from model.cubenet import CubeNet
from utils.get_tasks import get_tasks
from utils.opts import parse_opts
from keras import backend as K
import tensorflow as tf
from tensorflow import lite
# from utils.convert_model_to_NWHC import convert_frozen_model_to_NWHC


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph
    with graph.as_default():
        freeze_var_names = \
            list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


opt = parse_opts()

input_np = np.random.uniform(0, 1, (1, 3, opt.person_size, opt.person_size))
input_var = Variable(torch.FloatTensor(input_np))

attr, _ = get_tasks(opt)
model = CubeNet(opt.train, opt.conv, attr, pretrained=False, img_size=opt.person_size,
                attention=opt.attention, dropout=opt.dropout, at=opt.at, at_loss=opt.at_loss)

path = "CubeModel/pretrained/save_60.pth"
state_dict = torch.load(path, map_location='cpu')["state_dict"]
for k in list(state_dict.keys()):
    k_new = k[7:]
    state_dict[k_new] = state_dict[k]
    state_dict.pop(k)
model.load_state_dict(state_dict, strict=True)
model.eval()

# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(3, opt.person_size, opt.person_size,)],
                           change_ordering=True, verbose=True, name_policy='short')

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in k_model.outputs])
tf.train.write_graph(frozen_graph, ".", "my_model.pb", as_text=False)
# print([i for i in k_model.outputs])
# keras_file = "my_model.h5"
# keras.models.save_model(model, keras_file)
# converter = lite.TFLiteConverter.from_keras_model_file(keras_file)
# converter = lite.TFLiteConverter.from_keras_model(k_model)
# convert_frozen_model_to_NWHC("my_model.pb")
input_array = ['input_0']
output_array = ['output_0', 'output_1', 'output_2', 'output_3', 'output_4']
converter = lite.TFLiteConverter.from_frozen_graph("my_model.pb", input_array, output_array)

# tflite_model = converter.convert()
# open("my_model.tflite", "wb").write(tflite_model)

