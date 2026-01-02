from __future__ import print_function
import os
import time
import yaml
import random
import shutil
import argparse

import importlib

import torch.nn as nn
import numpy as np

from utils import *

from model.lep import LinearClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from engine_pretrain import pretrain
from engine_lep import train_lep, val_lep

work_dir = "./output_dir/"

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def import_class(name):
    # components = name.split('.')
    # print(components)
    # mod = __import__(components[0])  # import return model
    # for comp in components[1:]:
    #     mod = getattr(mod, comp)
    # return mod

    components = name.split('.')
    module_path = '.'.join(components[:-1])
    class_name = components[-1]

    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_parser():
    parser = argparse.ArgumentParser(description='SSCT')
    parser.add_argument(
        '--work-dir',
        type = str,
        default = "./output_dir/",
        help = 'where all run files will be kept'
    )
    parser.add_argument(
        '--config',
        default='./config/ntu_xsub.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--model-args',
        default=dict(),
        help='the settings for the transformer')
    parser.add_argument(
        '--model',
        type=str,
        default="model.ssctransformer.SegmentedSplineCoeffTransformer",
        help='between SSCoeffT or SSCoordT')
    parser.add_argument(
        '--base-transformer',
        type=str,
        default="model.transformers.CoordTransformerBase",
        help='Base model without decoder'
    )
    parser.add_argument(
        '--weights-transformer-path',
        type=str,
        default="./pre-trained/ntu_xsub.pt",
        help='the path to the weights of the transformer')
    parser.add_argument(
        '--weights-lep-path',
        type=str,
        default="./pre-trained/",
        help='the path to the weights of the LEP')
    parser.add_argument(
        '--train',
        type=str2bool,
        default=True,
        help='Perform pretraining?')
    parser.add_argument(
        '--train-lep',
        type=str2bool,
        default=True,
        help='Perform LEP?')
    parser.add_argument(
        '--base-lr-pretrain',
        type=float,
        default=0.001,
        help='initial learning rate for pretraining')
    parser.add_argument(
        '--base-lr-lep',
        type=float,
        default=0.1,
        help='initial learning rate for LEP')
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=20,
        help='maximum iteration before switching to Cosine Anneling LR Scheduler for pretraining')
    parser.add_argument(
        '--eta-min-pretrain',
        type=float,
        default=5e-4,
        help='minimum lr that the Cosine Annealing lr scheduler will minimize to for pre-training')
    parser.add_argument(
        '--weight-decay-pretrain',
        type=float,
        default=0.05,
        help='weight decay for AdamW optimizer during pretraining')
    parser.add_argument(
        '--weight-decay-lep',
        type=float,
        default=0.0,
        help='weight decay for AdamW optimizer during pretraining')
    parser.add_argument(
        '--epochs-pretrain',
        type=int,
        default=400,
        help='number of epochs during pretraining')
    parser.add_argument(
        '--epochs-lep',
        type=int,
        default=100,
        help='number of epochs during LEP')
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='seed used for reproducibility')
    parser.add_argument(
        '--save-weights',
        type=str2bool,
        default=True,
        help='to save the best weights or not')
    parser.add_argument(
        '--print-log-bool',
        type=str2bool,
        default=True,
        help='to print in the log or not')
    parser.add_argument(
        '--batch-size-pretrain',
        type=int,
        default=2**7,
        help='batch size for pretraining')
    parser.add_argument(
        '--batch-size-lep',
        type=int,
        default=2**6,
        help='batch size for pretraining')
    parser.add_argument(
        '--pretrain-feeder-args',
        default=dict(),
        help='the arguments of data loader for training'
    )
    parser.add_argument(
        '--mask-ratio',
        type = float,
        default = 0.8,
        help = 'embedding mask ratio'
    )
    parser.add_argument(
        '--lep-train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training'
    )
    parser.add_argument(
        '--lep-val-feeder-args',
        default=dict(),
        help='the arguments of data loader for training'
    )
    parser.add_argument(
        '--labels',
        type = list,
        default = None,
        help='Labels to generate the confusion matrix'
    )
    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='The data loader that will be used')
    parser.add_argument(
        '--num-classes',
        default=60,
        help='Number of Classes in Dataset for LEP')
    parser.add_argument(
        '--amp',
        type=str2bool,
        default=False,
        help='to activate mixed precision'
    )
    parser.add_argument(
        '--average-pool',
        type=str2bool,
        default=False,
        help='to use average pooled features'
    )
    parser.add_argument(
        '--per-class-acc',
        type=str2bool,
        default=False,
        help='to output per class accuracy'
    )

    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()

    # load args from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.full_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    init_seed(args.seed)

    global seed
    seed = args.seed
    global work_dir
    if args.work_dir is not None:
        work_dir = args.work_dir

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok = True)

    Feeder = import_class(args.feeder)

    dataset_train = Feeder(**args.pretrain_feeder_args)

    Model = import_class(args.model)

    try:
        model = Model(**args.model_args).cuda()
    except:
        print_log("Cuda not available", work_dir)
        model = Model(**args.model_args)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    print_log(f"Seed: {args.seed} \n ", work_dir)

    if args.train:
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size_pretrain,
            shuffle=True
        )

        # Set Bias and Norm Layers to 0 weight decay
        decay = []
        no_decay = []
        for name, s in model.named_parameters():
            if name.endswith('.bias') or any(norm in name for norm in ['norm', 'bn']):
                # print(name)
                no_decay.append(s)
            else:
                decay.append(s)

        segment_param = [
            {'params': decay, 'weight decay': args.weight_decay_pretrain},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(segment_param, lr=args.base_lr_pretrain, betas=(0.9, 0.95))

        def linear_warmup(step):
            if step >= args.warmup_epochs:
                return 1.0
            return step / args.warmup_epochs

        scheduler_warmup = LambdaLR(optimizer, lr_lambda=linear_warmup)
        scheduler_cos = CosineAnnealingLR(optimizer, T_max=(args.epochs_pretrain - args.warmup_epochs),
                                          eta_min=args.eta_min_pretrain)

        print_log(f"Pre-training Details: \nModel: {args.model}\nModel Settings: {args.model_args} \nMask Ratio: {args.mask_ratio} \nBatch Size: {args.batch_size_pretrain} \nEpochs: {args.epochs_pretrain}\nWarmup epochs: {args.warmup_epochs}", work_dir)

        print_log(f"Base LR: {args.base_lr_pretrain} \nETA Min: {args.eta_min_pretrain}\nWeight Decay:{args.weight_decay_pretrain}", work_dir)

        print_log(f"Pre-train Feeder Settings: {args.pretrain_feeder_args}", work_dir)

        pretrain(train_loader, model, optimizer, args.epochs_pretrain, args.model_args["segments"], args.mask_ratio, args.warmup_epochs, scheduler_warmup, scheduler_cos, work_dir=args.work_dir, amp = args.amp)
    BaseModel = import_class(args.base_transformer)
    model = BaseModel(**args.model_args).cuda()
    model = load_weights(model, args.weights_transformer_path, args.work_dir)
    model.eval()
    try:
        if not args.average_pool:
            classifier = LinearClassifier(args.model_args["d_model"] * (len(args.model_args["segments"])+len(args.model_args["hand_segments"])) * (args.lep_train_feeder_args["window_size"]//args.model_args["t_m"]), args.num_classes).cuda()
            print_log("Model has hand segments as embeddings", work_dir)
    except KeyError:
        if not args.average_pool:
            if not "3D" in args.model:
                try:
                    if not args.model_args["body_avg"]:
                        classifier = LinearClassifier(2*args.model_args["d_model"] * len(args.model_args["segments"]) * (
                                    args.lep_train_feeder_args["window_size"] // args.model_args["t_m"]),
                                                      args.num_classes).cuda()
                        print_log(f'Number of dimensions: {2 * args.model_args["d_model"] * len(args.model_args["segments"]) * (args.lep_train_feeder_args["window_size"] // args.model_args["t_m"])}', work_dir)
                    else:
                        classifier = LinearClassifier(args.model_args["d_model"] * len(args.model_args["segments"]) * (
                                    args.lep_train_feeder_args["window_size"] // args.model_args["t_m"]),
                                                      args.num_classes).cuda()
                except KeyError:
                    classifier = LinearClassifier(args.model_args["d_model"] * len(args.model_args["segments"]) * (args.lep_train_feeder_args["window_size"]//args.model_args["t_m"]), args.num_classes).cuda()
            else:
                classifier = LinearClassifier(3 * args.model_args["d_model"] * len(args.model_args["segments"]) * (
                            args.lep_train_feeder_args["window_size"] // args.model_args["t_m"]),
                                              args.num_classes).cuda()
                print_log("Model uses 3D embeddings", work_dir)
        else:
            classifier = LinearClassifier(args.model_args["d_model"], args.num_classes).cuda()

    if args.train_lep:
        classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=args.base_lr_lep, momentum=0.9,
                                               weight_decay=args.weight_decay_lep)
        scheduler_classifier = CosineAnnealingLR(classifier_optimizer, T_max=args.epochs_lep)
        dataset_train = Feeder(**args.lep_train_feeder_args)
        dataset_val = Feeder(**args.lep_val_feeder_args)
        classifier_criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size_lep,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_lep,
            shuffle=False
        )

        print_log(f"LEP Details: \nModel: {args.model} \n Model Settings: {args.model_args}\n LEP Train Feeder Settings: {args.lep_train_feeder_args} \n LEP Val Feeder Settings: {args.lep_val_feeder_args} \n Number of Classes: {args.num_classes} \nTransformer Weights Used: {args.weights_transformer_path}", work_dir)
        print_log(f"Batch Size: {args.batch_size_lep}\nEpochs:{args.epochs_lep}\nBase LR:{args.base_lr_lep}\nWeight Decay: {args.weight_decay_lep}", work_dir)
        print_log("Training Linear Classifier for LEP: ", work_dir)

        train_lep(train_loader, test_loader, model, classifier, classifier_optimizer, classifier_criterion,
              scheduler_classifier, args.epochs_lep, args.model_args["segments"], args.mask_ratio, amp=args.amp, work_dir=args.work_dir, labels=args.labels, avg_pool=args.average_pool)

    else:
        print_log(f"Transformer Weights Used: {args.weights_transformer_path} \nLEP Weights Used: {args.weights_lep_path}",work_dir)
        dataset_val = Feeder(**args.lep_val_feeder_args)

        test_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_lep,
            shuffle=False
        )

        val_lep(test_loader, model, classifier, args.model_args["segments"], args.mask_ratio, args.weights_lep_path, amp=args.amp, work_dir=args.work_dir, labels=args.labels, per_class_acc=args.per_class_acc)

if __name__ == '__main__':
    main()