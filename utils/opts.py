import argparse
from data.attributes import WiderAttributes as WdAt
from data.attributes import NewAttributes as NwAt
from data.attributes import BerkeleyAttributes as BkAt
import pretrainedmodels
import os
import datetime


# store_true set a switch to true while store_false set a switch to false
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='/root/dataset/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--result_path',
        default='log/',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='Wider',
        type=str,
        choices=['Wider', 'Berkeley', 'New'],
        help='(Wider). Multiple datasets can be specified using comma as separator')
    # only support wider dataset now
    parser.add_argument(
        '--mode',
        default='paper',
        type=str,
        choices=['paper', 'branch'])
    parser.add_argument(
        '-lr',
        '--lr',
        default=3e-3,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum')
    parser.add_argument(
        '-wd',
        '--weight_decay',
        default=1e-4,
        type=float,
        help='Weight Decay')
    parser.add_argument(
        '-do',
        '--dropout',
        default=0.1,
        type=float,
        help='dropout')
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='sgd',
        type=str,
        help='Currently support SGD and Adam')
    parser.add_argument(
        '--betas',
        default=(0.9, 0.99),
        nargs=2,
        type=float,
        help='Currently only support SGD')
    parser.add_argument(
        '-lp',
        '--lr_patience',
        default=5,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument(
        '-f',
        '--factor',
        default=0.1,
        type=float,
        help='Weight decay used to decay lr when on plateau')
    parser.add_argument(
        '-bs',
        '--batch_size',
        default=256,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '-ps',
        '--person_size',
        default=224,
        type=int,
        help='Size of face bounding box to be sized to')
    parser.add_argument(
        '-ne',
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    # parser.add_argument(
    #     '--begin_epoch',
    #     default=1,
    #     type=int,
    #     help='Training begins at this epoch. Previous trained models indicated by resume_path is loaded.'
    # )
    parser.add_argument(
        '--checkpoint',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '-pt',
        '--pretrain',
        action='store_true',
        help='Whether to use pretrained weights in conv models')
    parser.set_defaults(pretrain=True)
    # parser.add_argument(
    #     '--ft_begin_index',
    #     default=0,
    #     type=int,
    #     help='Begin block index of fine-tuning')
    parser.add_argument(
        '-nt',
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '-si',
        '--save_interval',
        default=60,
        type=int,
        help='Trained models is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='cube',
        type=str,
        choices=['cube'],
        help='just support cube now')
    parser.add_argument(
        '--conv',
        default='resnet18',
        choices=['resnet18', 'vgg', 'fpn18', 'detnet', 'mobilenet', 'mobile3l', 'mobile3s'],
        type=str,
        help=pretrainedmodels.model_names)
    parser.add_argument(
        '-cl',
        '--categorical_loss',
        default='cross_entropy',
        type=str,
        help='ohem | cross_entropy | focal | cross_entropy_weight'
    )
    parser.add_argument(
        '-sa',
        '--specified_attrs',
        default=[],
        type=str,
        nargs='+',
        choices=[*WdAt.names()] + [*NwAt.names()] + [*BkAt.names()],
        help='Currently only support Wider attr dataset')
    parser.add_argument(
        '-or',
        '--output_recognizable',
        action='store_true',
        help='Whether to also classify recognizability of each attribute')
    parser.add_argument(
        '-sra',
        '--specified_recognizable_attrs',
        default=[],
        # default=[*WdAt.names()],
        type=str,
        nargs='+',
        choices=[*WdAt.names()] + [*NwAt.names()] + [*BkAt.names()],
        help='Currently only support Wider attr dataset')
    parser.add_argument(
        '-ls',
        '--label_smooth',
        action='store_true',
        help='Whether to open the label smooth'
    )
    parser.add_argument(
        '-at',
        '--attention',
        default='Custom',
        type=str,
        # choices=['NoAttention', 'ScOd', 'OvFc', 'RfMp', 'SpRl', 'Super', 'PrTp', 'CamOvFc', 'Custom',
        #          'TwoLevel', 'ThreeLevel', 'TwoLevelAuto', 'ThreeLevelRNN', 'NoAttentionMuti', 'CPrTp', 'PCPrTp'],
        choices=['NoAttention', 'OvFc', 'PrTp', 'CPrTp', 'PCPrTp', 'NoAttentionForTrace'],
        help='Whether to add attention mechanism of each attribute'
             'ScOd: second-order pooling'
             'OvFc: all over you face'
             'RfMp: refining attention heat map'
             'SpRl: spatial regularization'
             'Super: mul-scale attention')
    # parser.add_argument(
    #     '-mn',
    #     '--map_norm',
    #     action='store_true',
    #     help='Norm the map to make sum of it become 1')
    parser.add_argument(
        '-li',
        '--log_interval',
        type=int,
        default=20,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '-ld',
        "--log_dir",
        type=str,
        help="log directory for Tensorboard log output")
    parser.add_argument(
        '-lf',
        "--log_file",
        default='logfile',
        type=str,
        help="log name")
    # parser.add_argument(
    #     '--no_train',
    #     action='store_true',
    #     help='If true, training is not performed.')
    # parser.set_defaults(no_train=False)
    parser.add_argument(
        '-at_sw',
        '--at',
        action='store_true',
        help='Whether to add attention loss')
    parser.add_argument(
        '--at_loss',
        default='MSE',
        type=str,
        choices=['MSE', 'KL'])
    parser.add_argument(
        '--at_coe',
        default=1.,
        type=float,
        help='coe for attention loss')
    parser.add_argument(
        '-st',
        '--state',
        action='store_true',
        help='Whether to test mode')
    parser.add_argument(
        '-al',
        '--at_level',
        default='wide',
        type=str,
        choices=['wide', 'thin', 'wrong'])
    parser.add_argument(
        '--img_path',
        default='',
        type=str,
        help='test img path')
    parser.add_argument(
        '--model_path',
        default='/root/dataset/save_60.pth',
        type=str,
        help='pretrained model path')
    parser.add_argument(
        '-tm',
        '--test_mode',
        default='pic',
        type=str,
        choices=['pic', 'train_dir', 'camera'],
        help='pic: a single picture,  train_dir: the train set and val set,  camera: for the real time test')
    args = parser.parse_args()
    if args.log_dir:
        args.log_dir = os.path.join(args.result_path, args.log_dir)
    else:
        args.log_dir = os.path.join(
            args.result_path, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not args.specified_attrs:
        if args.dataset == 'Wider':
            args.specified_attrs = WdAt.names()
        elif args.dataset == 'New':
            args.specified_attrs = NwAt.names()
        elif args.dataset == 'Berkeley':
            args.specified_attrs = BkAt.names()
        else:
            # TODO
            pass
    # else:
    #     args.specified_attrs = args.specified_attrs.strip().split()
    if args.output_recognizable:
        if not args.specified_recognizable_attrs:
            if args.dataset == 'Wider':
                args.specified_recognizable_attrs = WdAt.names()
            elif args.dataset == 'New':
                args.specified_recognizable_attrs = NwAt.names()
            elif args.dataset == 'Berkeley':
                args.specified_recognizable_attrs = BkAt.names()
            else:
                # TODO
                pass
        # else:
        #     args.specified_recognizable_attrs = args.specified_recognizable_attrs.strip().split()
    return args
