import torch
from utils.opts import parse_opts
from utils.get_tasks import get_tasks
from model.generate_model import generate_model

opt = parse_opts()
attr, _ = get_tasks(opt)
model, _, _, _ = generate_model(opt, attr)

state_dict = torch.load('../log/vgg16_new/save_1.pth')['state_dict']
model.load_state_dict(state_dict)
model.cpu()
example = torch.rand(1, 3, 224, 224)
a = torch.jit.trace(model.eval(), example)
a.save('my_test_cpu.pt')
