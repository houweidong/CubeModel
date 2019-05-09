from data.attributes import WiderAttributes as WdAt
from data.attributes import Attribute
from collections import OrderedDict


def get_tasks(opt):
    datasets = opt.dataset.split(",")

    # Get all available attributes from these datasets
    available_attrs = OrderedDict()
    for ds in datasets:
        # TODO Refactor to remove this if-else statement
        if ds == 'Wider':
            # according to opt.specified_attrs, opt.specified_recognizable_attrs
            # and opt.output_recognizable generate tasks
            attrs_ds = WdAt.list_attributes(opt)
        else:
            raise Exception("Not supported dataset {}".format(ds))

        for attr in attrs_ds:
            if attr.key not in available_attrs:
                available_attrs[attr.key] = attr
            else:
                # Merge attributes from different datasets
                available_attrs[attr.key] = available_attrs[attr.key].merge(attr)
    attributes = list(available_attrs.values())

    names = []
    for attr in attributes:
        assert isinstance(attr, Attribute)
        names.append(attr.name)
        if attr.rec_trainable:
            names.append(attr.name + '/recognizable')

    return attributes, names