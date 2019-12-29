# -*- coding: utf-8 -*-
import logging
import os

import torch
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip3 install tensorboardX")

from ignite.handlers import Timer
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from utils.opts import parse_opts
from model.generate_model import generate_model
from data.get_data import get_data
from utils.get_tasks import get_tasks
from utils.table import print_summar_table
from utils.logger import Logger
from utils.my_engine import my_trainer

from training.loss_utils import multitask_loss
from training.metric_utils import MultiAttributeMetric
from training.get_loss_metric import get_losses_metrics


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x.cuda())
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(opt):
    # logging.basicConfig(filename=os.path.join(opt.log_dir, opt.log_file), level=logging.INFO)
    # logger = logging.getLogger()
    # # logger.addHandler(logging.StreamHandler())
    # logger = logger.info

    log = Logger(filename=os.path.join(opt.log_dir, opt.log_file), level='debug')
    logger = log.logger.info

    # Decide what attrs to train
    attr, attr_name = get_tasks(opt)

    # Generate model based on tasks
    logger('Loading models')
    model, parameters, mean, std = generate_model(opt, attr)
    # parameters[0]['lr'] = 0
    # parameters[1]['lr'] = opt.lr / 3

    logger('Loading dataset')
    train_loader, val_loader = get_data(opt, attr, mean, std)
    writer = create_summary_writer(model, train_loader, opt.log_dir)
    # have to after writer
    model = nn.DataParallel(model, device_ids=None)
    # Learning configurations
    if opt.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = Adam(parameters, lr=opt.lr, betas=opt.betas)
    else:
        raise Exception("Not supported")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_patience, factor=opt.factor,
                                               min_lr=1e-6)

    # Loading checkpoint
    if opt.checkpoint:
        logger('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    device = 'cuda'
    loss_fns, metrics = get_losses_metrics(attr, opt.categorical_loss, opt.at, opt.at_loss)
    trainer = my_trainer(model, optimizer, lambda pred, target, epoch: multitask_loss(
        pred, target, loss_fns, len(attr_name), opt.at_coe, epoch), device=device)
    train_evaluator = create_supervised_evaluator(model, metrics={
        'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics={
        'multitask': MultiAttributeMetric(metrics, attr_name)}, device=device)

    # Training timer handlers
    model_timer, data_timer = Timer(average=True), Timer(average=True)
    model_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED,
                       step=Events.ITERATION_COMPLETED)
    data_timer.attach(trainer,
                      start=Events.EPOCH_STARTED,
                      resume=Events.ITERATION_COMPLETED,
                      pause=Events.ITERATION_STARTED,
                      step=Events.ITERATION_STARTED)

    # Training log/plot handlers
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter_num = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter_num % opt.log_interval == 0:
            logger("Epoch[{}] Iteration[{}/{}] Sum Loss: {:.2f} Cls Loss: {:.2f} At Loss: {:.2f} "
                   "Coe: {:.2f} Model Process: {:.3f}s/batch Data Preparation: {:.3f}s/batch".format(
                engine.state.epoch, iter_num, len(train_loader), engine.state.output['sum'],
                engine.state.output['cls'], engine.state.output['at'], engine.state.output['coe'],
                model_timer.value(), data_timer.value()))
            writer.add_scalar("training/loss", engine.state.output['sum'], engine.state.iteration)

    # Log/Plot Learning rate
    @trainer.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        lr = optimizer.param_groups[-1]['lr']
        logger('Epoch[{}] Starts with lr={}'.format(engine.state.epoch, lr))
        writer.add_scalar("learning_rate", lr, engine.state.epoch)

    # Checkpointing
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        if engine.state.epoch % opt.save_interval == 0:
            save_file_path = os.path.join(opt.log_dir, 'save_{}.pth'.format(engine.state.epoch))
            states = {
                'epoch': engine.state.epoch,
                'arch': opt.model,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            # model.eval()
            # example = torch.rand(1, 3, 224, 224)
            # traced_script_module = torch.jit.trace(model, example)
            # traced_script_module.save(save_file_path)
            # model.train()
            # torch.save(model._modules.state_dict(), save_file_path)

    # val_evaluator event handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        data_list = [train_loader, val_loader]
        name_list = ['train', 'val']
        eval_list = [train_evaluator, val_evaluator]

        for data, name, evl in zip(data_list, name_list, eval_list):
            evl.run(data)
            metrics_info = evl.state.metrics["multitask"]

            for m, val in metrics_info['metrics'].items():
                writer.add_scalar(name + '_metrics/{}'.format(m), val, engine.state.epoch)

            for m, val in metrics_info['summaries'].items():
                writer.add_scalar(name + '_summary/{}'.format(m), val, engine.state.epoch)

            logger(name + ": Validation Results - Epoch: {}".format(engine.state.epoch))
            print_summar_table(logger, attr_name, metrics_info['logger'])

            # Update Learning Rate
            if name == 'train':
                scheduler.step(metrics_info['logger']['attr']['ap'][-1])

    # kick everything off
    logger('Start training')
    trainer.run(train_loader, max_epochs=opt.n_epochs)

    writer.close()


if __name__ == "__main__":
    args = parse_opts()

    run(args)
