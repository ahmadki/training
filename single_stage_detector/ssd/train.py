import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import logging
from mlperf_logging.mllog import constants as mllog_const
from mlperf_logger import ssd_print, broadcast_seeds
from mlperf_logger import mllogger

_BASE_LR=2.5e-3

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help='path to pretrained backbone weights file, '
                             'default is to get it from online torchvision repository')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each training iteration')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='number of examples for each validation iteration (defaults to --batch-size)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='epoch interval for validation in addition to --val-epochs.')
    parser.add_argument('--val-epochs', nargs='*', type=int,
                        default=[],
                        help='epochs at which to evaluate in addition to --val-interval')
    parser.add_argument('--batch-splits', type=int, default=1,
                        help='Split batch to N steps (gradient accumulation)')
    parser.add_argument('--lr-decay-schedule', nargs='*', type=int,
                        default=[40, 50],
                        help='epochs at which to decay the learning rate')
    parser.add_argument('--warmup', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', type=int, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--lr', type=float, default=_BASE_LR,
                        help='base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay factor')
    parser.add_argument('--num-cropping-iterations', type=int, default=1,
                        help='cropping retries in augmentation pipeline, '
                             'default 1, other legal value is 50')
    parser.add_argument('--nms-valid-thresh', type=float, default=0.05,
                        help='in eval, filter input boxes to those with score greater '
                             'than nms_valid_thresh.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
    # Distributed stuff
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='Used for multi-process training. Can either be manually set '
                             'or automatically set by using \'python -m multiproc\'.')

    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def coco_eval(model, val_dataloader, cocoGt, encoder, inv_map, threshold,
              epoch, iteration, log_interval=100,
              use_cuda=True, nms_valid_thresh=0.05):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    print("nms_valid_thresh is set to {}".format(nms_valid_thresh))

    mllogger.start(
        key=mllog_const.EVAL_START,
        metadata={mllog_const.EPOCH_NUM: epoch})

    start = time.time()
    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
        with torch.no_grad():
            if use_cuda:
                img = img.cuda()
            ploc, plabel = model(img)

            try:
                results = encoder.decode_batch(ploc, plabel,
                                               overlap_threshold,
                                               nms_max_detections,
                                               nms_valid_thresh=nms_valid_thresh)
            except:
                #raise
                print("")
                print("No object detected in batch: {}".format(nbatch))
                continue

            (htot, wtot) = [d.cpu().numpy() for d in img_size]
            img_id = img_id.cpu().numpy()
            # Iterate over batch elements
            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                loc, label, prob = [r.cpu().numpy() for r in result]

                # Iterate over image detections
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id_, loc_[0]*wtot_, \
                                         loc_[1]*htot_,
                                         (loc_[2] - loc_[0])*wtot_,
                                         (loc_[3] - loc_[1])*htot_,
                                         prob_,
                                         inv_map[label_]])
        if log_interval and not (nbatch+1) % log_interval:
                print("Completed inference on batch: {}".format(nbatch+1))

    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]

    ssd_print(key=mllog_const.EVAL_ACCURACY,
              value=current_accuracy,
              metadata={mllog_const.EPOCH_NUM: epoch},
              sync=False)
    mllogger.end(
        key=mllog_const.EVAL_STOP,
        metadata={mllog_const.EPOCH_NUM: epoch})
    return current_accuracy >= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

def lr_warmup(optim, wb, iter_num, base_lr, args):
	if iter_num < wb:
		# mlperf warmup rule
		warmup_step = base_lr / (wb * (2 ** args.warmup_factor))
		new_lr = base_lr - (wb - iter_num) * warmup_step

		for param_group in optim.param_groups:
			param_group['lr'] = new_lr

def train300_mlperf_coco(args):
    global torch
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.distributed = False
    if use_cuda:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        except:
            raise ImportError("Please install APEX from https://github.com/nvidia/apex")

    local_seed = args.seed
    if args.distributed:
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
        if args.no_cuda:
            device = torch.device('cpu')
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda')
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device)
            local_seed = (args.seed + dist.get_rank()) % 2**32
    mllogger.event(key=mllog_const.SEED, value=local_seed)
    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)

    args.rank = dist.get_rank() if args.distributed else args.local_rank
    print("args.rank = {}".format(args.rank))
    print("local rank = {}".format(args.local_rank))
    print("distributed={}".format(args.distributed))

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False,
                                 num_cropping_iterations=args.num_cropping_iterations)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    mllogger.event(key=mllog_const.TRAIN_SAMPLES, value=len(train_coco))
    mllogger.event(key=mllog_const.EVAL_SAMPLES, value=len(val_coco))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_coco)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_coco,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=4)
    # set shuffle=True in DataLoader
    if args.rank==0:
        val_dataloader = DataLoader(val_coco,
                                    batch_size=args.val_batch_size or args.batch_size,
                                    shuffle=False,
                                    sampler=None,
                                    num_workers=4)
    else:
        val_dataloader = None

    ssd300 = SSD300(train_coco.labelnum, model_path=args.pretrained_backbone)
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
    ssd300.train()
    if use_cuda:
        ssd300.cuda()
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

	# parallelize
    if args.distributed:
        ssd300 = DDP(ssd300)

    global_batch_size = N_gpu * args.batch_size
    mllogger.event(key=mllog_const.GLOBAL_BATCH_SIZE, value=global_batch_size)
    # Reference doesn't support group batch norm, so bn_span==local_batch_size
    mllogger.event(key=mllog_const.MODEL_BN_SPAN, value=args.batch_size)
    current_lr = args.lr * (global_batch_size / 32)

    assert args.batch_size % args.batch_splits == 0, "--batch-size must be divisible by --batch-splits"
    fragment_size = args.batch_size // args.batch_splits
    if args.batch_splits != 1:
        print("using gradient accumulation with fragments of size {}".format(fragment_size))

    current_momentum = 0.9
    optim = torch.optim.SGD(ssd300.parameters(), lr=current_lr,
                            momentum=current_momentum,
                            weight_decay=args.weight_decay)
    ssd_print(key=mllog_const.OPT_BASE_LR, value=current_lr)
    ssd_print(key=mllog_const.OPT_WEIGHT_DECAY, value=args.weight_decay)

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}
    success = torch.zeros(1)
    if use_cuda:
        success = success.cuda()


    if args.warmup:
        nonempty_imgs = len(train_coco)
        wb = int(args.warmup * nonempty_imgs / (N_gpu*args.batch_size))
        ssd_print(key=mllog_const.OPT_LR_WARMUP_STEPS, value=wb)
        warmup_step = lambda iter_num, current_lr: lr_warmup(optim, wb, iter_num, current_lr, args)
    else:
        warmup_step = lambda iter_num, current_lr: None

    ssd_print(key=mllog_const.OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)
    ssd_print(key=mllog_const.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=args.lr_decay_schedule)
    mllogger.start(
        key=mllog_const.BLOCK_START,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    optim.zero_grad()
    for epoch in range(args.epochs):
        mllogger.start(
            key=mllog_const.EPOCH_START,
            metadata={mllog_const.EPOCH_NUM: epoch})
        # set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_decay_schedule:
            current_lr *= 0.1
            print("")
            print("lr decay step #{num}".format(num=args.lr_decay_schedule.index(epoch) + 1))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr

        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(train_dataloader):
            current_batch_size = img.shape[0]
            # Split batch for gradient accumulation
            img = torch.split(img, fragment_size)
            bbox = torch.split(bbox, fragment_size)
            label = torch.split(label, fragment_size)

            for (fimg, fbbox, flabel) in zip(img, bbox, label):
                current_fragment_size = fimg.shape[0]
                trans_bbox = fbbox.transpose(1,2).contiguous()
                if use_cuda:
                    fimg = fimg.cuda()
                    trans_bbox = trans_bbox.cuda()
                    flabel = flabel.cuda()
                fimg = Variable(fimg, requires_grad=True)
                ploc, plabel = ssd300(fimg)
                gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                               Variable(flabel, requires_grad=False)
                loss = loss_func(ploc, plabel, gloc, glabel)
                loss = loss * (current_fragment_size / current_batch_size) # weighted mean
                loss.backward()

            warmup_step(iter_num, current_lr)
            optim.step()
            optim.zero_grad()
            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()
            if args.rank == 0 and args.log_interval and not iter_num % args.log_interval:
                print("Iteration: {:6d}, Loss function: {:5.3f}, Average Loss: {:.3f}"\
                    .format(iter_num, loss.item(), avg_loss))
            iter_num += 1

        do_eval = (args.val_epochs and (epoch+1) in args.val_epochs) or \
                  (args.val_interval and not (epoch+1) % args.val_interval)
        do_eval = do_eval and (epoch+1)>=50
        if do_eval:
            if args.distributed:
                world_size = float(dist.get_world_size())
                for bn_name, bn_buf in ssd300.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size
                        ssd_print(key=mllog_const.MODEL_BN_SPAN,
                            value=bn_buf)
            if args.rank == 0:
                if not args.no_save:
                    print("")
                    print("saving model...")
                    torch.save({"model" : ssd300.state_dict(), "label_map": train_coco.label_info},
                               "./models/iter_{}.pt".format(iter_num))

                if coco_eval(ssd300, val_dataloader, cocoGt, encoder, inv_map,
                             args.threshold, epoch + 1, iter_num,
                             log_interval=args.log_interval,
                             nms_valid_thresh=args.nms_valid_thresh):
                    success = torch.ones(1)
                    if use_cuda:
                        success = success.cuda()
            if args.distributed:
                dist.broadcast(success, 0)
            if success[0]:
                    return True
            mllogger.end(
                key=mllog_const.EPOCH_STOP,
                metadata={mllog_const.EPOCH_NUM: epoch})
    mllogger.end(
        key=mllog_const.BLOCK_STOP,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    return False

def main():
    mllogger.start(key=mllog_const.INIT_START)
    args = parse_args()

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
            os.mkdir('./models')

    torch.backends.cudnn.benchmark = True

    # start timing here
    mllogger.end(key=mllog_const.INIT_STOP)
    mllogger.start(key=mllog_const.RUN_START)

    success = train300_mlperf_coco(args)

    # end timing here
    mllogger.end(key=mllog_const.RUN_STOP, value={"success": success})


if __name__ == "__main__":
    main()
