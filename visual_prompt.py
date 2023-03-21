from __future__ import print_function

import argparse, os, time, random
from tqdm import tqdm

import torch, torchvision
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import *

import clip
from models import prompters
from models.prompters import TokenPrompter
from models.model import *
from attacks import *

from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import load_train_dataset, load_val_datasets, get_text_prompts_train, \
    get_text_prompts_val

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser('Adapting CLIP for zero-shot adv robustness')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1,
                        help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,  ## Why so large
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=5)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'null_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=10,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--imagenet_root', type=str, default=None)

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--attack', choices=['pgd', 'CW'], default='pgd')
    parser.add_argument('--train_class_count', type=int, default=90)
    parser.add_argument('--noimginprop', action='store_true')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)
    return args
    

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    global best_acc1, device

    args = parse_option()
    args.train_eps = args.train_eps / 255.
    args.test_eps = args.test_eps / 255.
    args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = args.test_stepsize / 255.

    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    import socket
    if socket.gethostname() == 'cv12' or socket.gethostname() == 'cv13':
        args.imagenet_root = '/local/vondrick/chengzhi/ImageNet-clean'
    else:
        args.imagenet_root = '/local/*/datasets/ImageNet-clean'

    # create model
    add_prompt_len = args.add_prompt_size

    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)  # .to(device)
    model.eval()

    prompter = prompters.__dict__[args.method](args)
    add_prompter = TokenPrompter(add_prompt_len)

    prompter = torch.nn.DataParallel(prompter).to(device)
    add_prompter = torch.nn.DataParallel(add_prompter).to(device)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            if 'vision_encoder_state_dict' in checkpoint.keys():
                # Load backbone for complementary experiment,
                # this assume that the finetuned model does not have the following prompts
                model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'], strict=False)
            else:
                # load only prompts, not backbone
                prompter.load_state_dict(checkpoint['state_dict'])
                add_prompter.load_state_dict(checkpoint['add_prompter'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')


    # load training dataset
    train_dataset = load_train_dataset(args)

    # load val dataset(s)
    if args.evaluate:
        val_dataset_name = ['cifar10', 'cifar100', 'STL10', 'SUN397', 'StanfordCars',  'Food101',
                            'oxfordpet', 'flowers102', 'Country211', 'dtd', 'EuroSAT', 'fgvc_aircraft',
                            'PCAM', 'hateful_memes', 'ImageNet', 'Caltech101', 'Caltech256']
    else:
        val_dataset_name = ['cifar10', 'cifar100',  'dtd', 'EuroSAT',]

    val_dataset_list = load_val_datasets(args, val_dataset_name)

    # create dataloaders
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler) for each in
                       val_dataset_list]

    # get text prompts for training/val
    texts_train = get_text_prompts_train(args, train_dataset, template=template)
    texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)


    # define criterion and optimizer
    optimizer = torch.optim.SGD(list(prompter.parameters()) + list(add_prompter.parameters()),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    if args.evaluate:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,
                                 prompter, add_prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, texts_train, model, prompter, add_prompter, optimizer, scheduler,
              criterion, scaler, epoch, args)

        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model,
                                 prompter, add_prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break


def train(train_loader, texts, model, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()
    add_prompter.train()

    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # print('bs', BATCH_SIZE)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # print(images.min(), images.max())

        # with automatic mixed precision
        with autocast():
            if not args.VPbaseline:
                delta = attack_pgd(prompter, model, add_prompter, criterion, images,
                                target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=args.train_eps)
                tmp = clip_img_preprocessing(images + delta)
            else:
                tmp = clip_img_preprocessing(images)

            prompted_images = prompter(tmp)
            prompt_token = add_prompter()

            # for multiple GPU
            output, _ = multiGPU_CLIP(model, prompted_images, text_tokens, prompt_token)

            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


# def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
def validate(val_loader_list, val_dataset_name, texts_list, model,
                prompter, add_prompter, criterion, args):
    dataset_num = len(val_loader_list)
    acc_all = []

    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()

        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if 'cifar' not in val_dataset_name:
                if i % 20 != 0 and not args.evaluate:
                    continue
            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    prompt_token = add_prompter()
                    output_prompt, _ = multiGPU_CLIP(model, prompter(clip_img_preprocessing(images)),
                                                text_tokens, prompt_token)

                    output_org, _ = multiGPU_CLIP(model, clip_img_preprocessing(images),
                                                  text_tokens, None)

                    loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()

                # generate adv example
                if args.attack == 'CW':
                    delta_prompt = attack_CW(prompter, model, add_prompter, criterion, images, target, text_tokens,
                                          test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                else:
                    delta_prompt = attack_pgd(prompter, model, add_prompter, criterion, images, target, text_tokens,
                                          test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    output_prompt_adv, _ = multiGPU_CLIP(model, prompter(clip_img_preprocessing(images + delta_prompt)),
                                                            text_tokens, prompt_token)
                    loss = criterion(output_prompt_adv, target)

                # bl attack
                torch.cuda.empty_cache()

                if args.attack == 'CW':
                    delta_noprompt = attack_CW(None, model, None, criterion, images, target, text_tokens,
                                          test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                else:
                    delta_noprompt = attack_pgd(None, model, None, criterion, images, target, text_tokens,
                                          test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    output_org_adv, _ = multiGPU_CLIP(model, clip_img_preprocessing(images + delta_noprompt),
                                                      text_tokens, None)

                torch.cuda.empty_cache()
                # measure accuracy and record loss
                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_prompt.update(acc1[0].item(), images.size(0))

                acc1 = accuracy(output_org_adv, target, topk=(1,))
                top1_adv_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.debug:
                    break

        torch.cuda.empty_cache()

        print(dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
                      top1_prompt=top1_prompt, top1_org=top1_org))
        acc_all.append(top1_adv_prompt.avg)

    return np.mean(acc_all)


if __name__ == '__main__':
    main()
