import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
from model_utils.data import getDataLoader
from model_utils.model import model_Datapara
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def main(device, args):

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(train=True, **args.aug_kwargs),
    #         train=True,
    #         **args.dataset_kwargs),
    #     shuffle=True,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )
    # memory_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
    #         train=True,
    #         **args.dataset_kwargs),
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=get_dataset(
    #         transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
    #         train=False,
    #         **args.dataset_kwargs),
    #     shuffle=False,
    #     batch_size=args.train.batch_size,
    #     **args.dataloader_kwargs
    # )+
    save_path = 'save_1.model'
    model = get_model(args.model)
    model = model_Datapara(model, device,save_path)
    model = torch.nn.DataParallel(model)
    filepath = '/home/lhy/hw3/food-11'
    train_loader = getDataLoader(filepath, 'train_unl', True, args.train.batch_size, transform=get_aug(train=True, train_classifier=False, **args.aug_kwargs))
    memory_loader = getDataLoader(filepath, 'train', False,args.train.batch_size, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    test_loader = getDataLoader(filepath, 'val', False,args.train.batch_size, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    # # semi_loader = getDataLoader(filepath,'train_unl', args.train.batch_size, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
    # # define model


    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr*args.train.batch_size/256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256,
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    max_acc = 0
    for epoch in global_progress:
        model.train()

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2)) in enumerate(local_progress):

            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            # logger.update_scalers(data_dict)


        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)
        if accuracy > max_acc:
            torch.save(model, 'save_1.model')
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
    #     logger.update_scalers(epoch_dict)
    #
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = save_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()
    args.aug_kwargs['image_size'] = 224
    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














