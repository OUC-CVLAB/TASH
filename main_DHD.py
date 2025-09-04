from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import *
from Dataloader import Loader, ImageList
from Retrieval import DoRetrieval
import scipy.io as sio
import os
import torch.distributed as dist
from config import config, update_from_args
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('DHD', add_help=False)

    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")  # qss
    # parser.add_argument('--pretrained_dir', default="/mnt/8TDisk1/zhenglab/lbq/sam_ViT-B_16.npz", type=str)
    # parser.add_argument('--data_dir', default="/mnt/8TDisk1/zhenglab/dataset/WildFish/", type=str,
    #                     help="""Path to dataset.""")  # qss
    # parser.add_argument('--train_txt', default="/mnt/8TDisk1/zhenglab/lbq/DHD/data/wildfish_Train.txt", type=str,
    #                     help="""Path to train txt file.""")  # qss
    # parser.add_argument('--db_txt', default="/mnt/8TDisk1/zhenglab/lbq/DHD/data/wildfish_DB.txt", type=str,
    #                     help="""Path to db txt file.""")  # qss
    # parser.add_argument('--query_txt', default="/mnt/8TDisk1/zhenglab/lbq/DHD/data/wildfish_Query.txt", type=str,
    #                     help="""Path to query txt file.""")
    parser.add_argument('--pretrained_dir', default="/mnt/8TDisk1/zhenglab/buguanren/sam_ViT-B_16.npz", type=str)
    parser.add_argument('--data_dir', default="/mnt/8TDisk1/zhenglab/dataset/WildFish/", type=str,
                        help="""Path to dataset.""")  # qss
    parser.add_argument('--train_txt', default="/mnt/8TDisk1/zhenglab/buguanren/DHD/data/wildfish_Train.txt", type=str,
                        help="""Path to train txt file.""")  # qss
    parser.add_argument('--db_txt', default="/mnt/8TDisk1/zhenglab/buguanren/DHD/data/wildfish_DB.txt", type=str,
                        help="""Path to db txt file.""")  # qss
    parser.add_argument('--query_txt', default="/mnt/8TDisk1/zhenglab/buguanren/DHD/data/wildfish_Query.txt", type=str,
                        help="""Path to query txt file.""")  # qss
    # parser.add_argument('--train_txt', default="/mnt/8TDisk1/zhenglab/dataset/sea_animals/test-agreement/agreement2/one-hot/train.txt", type=str,
    #                     help="""Path to train txt file.""")  # qss
    # parser.add_argument('--db_txt', default="/mnt/8TDisk1/zhenglab/dataset/sea_animals/test-agreement/agreement2/one-hot/database.txt", type=str,
    #                     help="""Path to db txt file.""")  # qss
    # parser.add_argument('--query_txt', default="/mnt/8TDisk1/zhenglab/dataset/sea_animals/test-agreement/agreement2/one-hot/test.txt", type=str,
    #                     help="""Path to query txt file.""")  # qss
    parser.add_argument('--dataset', default="wildfish", type=str, help="""Dataset name: imagenet, nuswide_m, coco, etc.""")  # qss
    
    parser.add_argument('--batch_size', default=64, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=12, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--encoder', default="ViT", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")  # qss
    parser.add_argument('--N_bits', default=32, type=int, help="""Number of bits to retrieval.""")  # qss
    parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.(default:3e-4)""")
    parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--transformation_scale', default=0.2, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")

    parser.add_argument('--max_epoch', default=150, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--eval_epoch', default=1, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--eval_init', default=1, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--amp', default=True, type=bool)
    parser.add_argument('--single_gpu', default=True, type=bool)
    parser.add_argument('--output_dir', default=f"./model", type=str, help="""Path to save logs and checkpoints.""")  # qss, TODO
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume .pt checkpoint')
    parser.add_argument('--seed', default=123, type=int, help="Random seed.")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help="weight_decay.")
    parser.add_argument('--test_log', default='./logs/final(32bit)_test_map.txt', type=str,
                        help="Path to save test mAP log.")
    parser.add_argument('--tsne_vis', action='store_true', help="Run t-SNE visualization on test set")

    return parser


class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.Hash = nn.Sequential(
            nn.Linear(fc_dim, N_bits, bias=False),
            nn.LayerNorm(N_bits))
        self.P = nn.Parameter(T.FloatTensor(NB_CLS, N_bits), requires_grad=True)
        nn.init.xavier_uniform_(self.P, gain=nn.init.calculate_gain('tanh'))

    def forward(self, X):

        X = self.Hash(X)
        return T.tanh(X)


def train(args):
    set_seed(args.seed)
    #amp
    scaler = T.cuda.amp.GradScaler(enabled=args.amp)

    if not args.single_gpu:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main = dist.get_rank() == 0
    else:
        device = torch.device("cuda")
        is_main = True

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # device = T.device('cuda')
    path = args.data_dir
    dname = args.dataset  # qss

    N_bits = args.N_bits
    init_lr = args.init_lr
    batch_size = args.batch_size

    # args.output_dir = f"./model_1/amp-select3(learnable)-sigmoid(no)_sam_{args.dataset}-{args.encoder}-{args.N_bits}bit-bs{args.batch_size}-epoch{args.max_epoch}"
    # args.output_dir = f"./model_1/amp-merge-mlp-(5-6-7-8-9-10-11)_sam_{args.dataset}-{args.encoder}-{args.N_bits}bit-bs{args.batch_size}-epoch{args.max_epoch}"
    # args.output_dir = f"./model_1/amp-(11-mlp-12)_sam_{args.dataset}-{args.encoder}-{args.N_bits}bit-bs{args.batch_size}-epoch{args.max_epoch}"
    # args.output_dir = f"./model_1/amp-standard_sam_{args.dataset}-{args.encoder}-{args.N_bits}bit-bs{args.batch_size}-epoch{args.max_epoch}"
    args.output_dir = f"./model/try_AdamW_{args.weight_decay}_lr{args.init_lr}_amp_CLAHE0.2_Retinex0.2_White_Balance0.3_(5-6-7-8-9-10)_sam_{args.dataset}-{args.encoder}-{args.N_bits}bit-bs{args.batch_size}-epoch{args.max_epoch}"
    update_from_args(args)  # 拷贝args到config中

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    if dname=='imagenet':  # qss
        NB_CLS=100
        Top_N=1000
    elif dname=='nuswide':
        NB_CLS=21
        Top_N=5000
    elif dname=='nuswide_m':
        NB_CLS=21
        Top_N=5000
    elif dname=='coco':
        NB_CLS=80
        Top_N=5000
    elif dname=='imagenet-100c':
        NB_CLS=100
        Top_N=5000
    elif dname=='imagenet-150k':
        NB_CLS=1000
        Top_N=2000
    elif dname=='sun-120k':
        NB_CLS=397
        Top_N=3950
    elif dname=='sun-attributes':
        NB_CLS=102
        Top_N=3950
    elif dname=='wildfish':
        NB_CLS=685
        Top_N=5000
    elif dname=='sea_animals':
        NB_CLS=23
        Top_N=5000
    else:
        print("Wrong dataset name.")
        return

    config.num_classes = NB_CLS
    config.hash_bit = args.N_bits

    # Img_dir = path+dname+'256'  # qss
    Img_dir = path
    # Train_dir = path+dname+'_Train.txt'  # qss
    Train_dir = args.train_txt
    # Gallery_dir = path+dname+'_DB.txt'  # qss
    Gallery_dir = args.db_txt
    # Query_dir = path+dname+'_Query.txt'  # qss
    Query_dir = args.query_txt

    org_size = 256
    input_size = 224
    
    AugS = Augmentation(org_size, 1.0, use_pre_enhance=True)
    AugT = Augmentation(org_size, args.transformation_scale, use_pre_enhance=True)

    Crop = nn.Sequential(Kg.CenterCrop(input_size))
    Norm = nn.Sequential(Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225])))

    trainset = ImageList(Img_dir, open(Train_dir).readlines())
    # trainloader = T.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
    #                                         shuffle=True, num_workers=args.num_workers)

    if not args.single_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True
    )

    # if is_main:
    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir)
    #     log_path = os.path.join(args.output_dir, "train_log.txt")
    #     with open(log_path, "w") as f:
    #         f.write("")  # 清空旧日志
    
    if args.encoder=='AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    elif args.encoder=='ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.encoder=='ViT':
        hash_bit=args.N_bits
        Baseline = VisionTransformer(config)
        Baseline.load_from(np.load(args.pretrained_dir))
        fc_dim = 768
    elif args.encoder=='DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.encoder=='SwinT':
        Baseline = SwinT('swin_base_patch4_window7_224')
        fc_dim = 1024
    else:
        print("Wrong dataset name.")
        return

    H = Hash_func(fc_dim, N_bits, NB_CLS)
    net = nn.Sequential(Baseline, H)
    # checkpoint = T.load(
    #     '/home/ouc/data1/qiaoshishi/python_codes/Deep-Hash-Distillation-main/models/imagenet-150k/imagenet-150k-checkpoint-512-model.pt')
    # state_dict = checkpoint['model_state_dict']
    # last_epoch = checkpoint['epoch']
    # net.load_state_dict(state_dict)

    net.cuda(device)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    if not args.single_gpu:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    HP_criterion = HashProxy(args.temp)
    HD_criterion = HashDistill()
    REG_criterion = BCEQuantization(args.std)

    params = [{'params': Baseline.parameters(), 'lr': 0.05*init_lr},
            {'params': H.parameters()}]

    optimizer = T.optim.Adam(params, lr=init_lr, weight_decay=args.weight_decay)
    # optimizer = T.optim.SGD(params, lr=init_lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0, last_epoch=-1) # last_epoch为开始训练的epoch

    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"=> loading checkpoint from '{args.resume_path}'")
        checkpoint = torch.load(args.resume_path)
        print("Keys in checkpoint:", checkpoint.keys())
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # lr_factors = [0.05, 1.0]
        # for param_group, factor in zip(optimizer.param_groups, lr_factors):
        #     if 'initial_lr' not in param_group:
        #         param_group['initial_lr'] = factor * init_lr

        scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0,last_epoch=checkpoint['epoch'] - args.warm_up)
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch - args.warm_up, eta_min=0, last_epoch=-1)
        if 'scheduler' in checkpoint:
            print("load scheduler from checkpoint")
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        MAX_mAP = checkpoint.get('best_map', 0.0)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0,last_epoch=-1)  # last_epoch为开始训练的epoch
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch - args.warm_up, eta_min=0, last_epoch=-1)
        print("=> no checkpoint loaded or file not found, training from scratch.")
        start_epoch = 0
        MAX_mAP = 0.0

    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("")

    mAP = 0.0
    # MAX_mAP = checkpoint['best_map']
    # qss tmp codes for extract codes offline
    # mAP, query_B, db_B, query_L, db_L = DoRetrieval(device, net.eval(), Img_dir, Gallery_dir, Query_dir,
    #                                                 NB_CLS, Top_N, args)  # qss
    # query_L = np.argmax(query_L.data.cpu().numpy(), 1).astype(np.uint32)
    # db_L = np.argmax(db_L.data.cpu().numpy(), 1).astype(np.uint32)
    # sio.savemat(os.path.join(args.output_dir,
    #                          args.dataset + '-' + str(args.N_bits) + '-' + str(mAP.item()) + "_offline.mat"),
    #             {'database_binary': db_B.data.cpu().numpy(), 'query_binary': query_B.data.cpu().numpy(),
    #              'database_label': db_L, 'query_label': query_L})

    for epoch in range(start_epoch, args.max_epoch):  # loop over the dataset multiple times
        if not args.single_gpu:
            trainloader.sampler.set_epoch(epoch)  # 免废根据 epoch 重排序
        if is_main:
            print('Epoch:', epoch, 'LR:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        C_loss = 0.0
        S_loss = 0.0
        R_loss = 0.0


        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            if labels.dim() == 1:
                labels = T.eye(NB_CLS, device=device)[labels]
            else:
                labels = labels.to(T.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            l1 = T.tensor(0., device=device)
            l2 = T.tensor(0., device=device)
            l3 = T.tensor(0., device=device)
            if args.amp:
                with T.cuda.amp.autocast(enabled=True):
                    Is = Norm(Crop(AugS(inputs))).to(device)
                    It = Norm(Crop(AugT(inputs))).to(device)
                    Xt = net(It)
                    l1 = HP_criterion(Xt, H.P, labels)
                    Xs = net(Is)
                    l2 = HD_criterion(Xs, Xt) * args.lambda1

                l3 = REG_criterion(Xt) * args.lambda2  # outside autocast
                loss = l1 + l2 + l3

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                Is = Norm(Crop(AugS(inputs))).to(device)
                It = Norm(Crop(AugT(inputs))).to(device)
                Xt = net(It)
                l1 = HP_criterion(Xt, H.P, labels)
                Xs = net(Is)
                l2 = HD_criterion(Xs, Xt) * args.lambda1
                l3 = REG_criterion(Xt) * args.lambda2
                loss = l1 + l2 + l3

                loss.backward()
                optimizer.step()

            # print statistics
            C_loss += l1.item()
            S_loss += l2.item()
            R_loss += l3.item()
            # reg_loss += l4.item()
            # T_loss += threshold_loss.item()

            # if (i+1) % 10 == 0:    # print every 10 mini-batches
            #     print('[%3d] C: %.4f, S: %.4f, R: %.4f, mAP: %.4f, MAX mAP: %.4f' %
            #         (i+1, C_loss / 10, S_loss / 10, R_loss / 10, mAP, MAX_mAP))
            #     C_loss = 0.0
            #     S_loss = 0.0
            #     R_loss = 0.0
        a = torch.sigmoid(net[0].transformer.encoder.a_logit)
        if epoch >= args.warm_up:
            scheduler.step()
        #log
        if is_main and (epoch + 1) % args.eval_epoch != 0:
            log_line = f"[Epoch {epoch + 1}] C_loss:{C_loss:.4f}, S_loss:{S_loss:.4f}, R_loss:{R_loss:.4f}"
            print(log_line)
            with open(log_path, "a") as f:
                f.write(log_line + "\n")

        if is_main and (epoch+1) % args.eval_epoch == 0 and (epoch+1) >= args.eval_init:
            mAP, query_B, db_B, query_L, db_L = DoRetrieval(device, net.eval(), Img_dir, Gallery_dir, Query_dir,
                                                            NB_CLS, Top_N, args)  # qss

            # if mAP > MAX_mAP or (epoch+1) % 100 == 0:  # qss, TODO, save model
            #     if mAP > MAX_mAP:
            if mAP > MAX_mAP:
                MAX_mAP = mAP
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                for file in os.listdir(args.output_dir):
                    if file.endswith(f"-{args.dataset}-checkpoint-{args.N_bits}-model.pt"):
                        os.remove(os.path.join(args.output_dir, file))

                    elif file.endswith(".mat") or file.endswith(".npz"):
                        if f"-{args.dataset}-{args.N_bits}-" in file:  # 确保是当前任务的文件
                            print(f"Removing old result file: {file}")
                            os.remove(os.path.join(args.output_dir, file))

                print("save in ", args.output_dir)
                T.save({'model_state_dict': net.state_dict(), 'epoch': epoch,
                        'optimizer': optimizer.state_dict(), 'best_map': MAX_mAP,'scheduler': scheduler.state_dict()},
                         os.path.join(args.output_dir, 'epoch-' + str(epoch+1) + '-' +
                                      args.dataset + '-' + 'checkpoint' + '-' + str(args.N_bits) + "-model.pt"))

                query_L = np.argmax(query_L.data.cpu().numpy(), 1).astype(np.uint32)
                db_L = np.argmax(db_L.data.cpu().numpy(), 1).astype(np.uint32)
                np.savez(
                    os.path.join(args.output_dir, 'epoch-' + str(epoch+1) + '-' + args.dataset + '-' + str(args.N_bits) + '-' + str(mAP.item()) + ".npz"),
                    database_binary=db_B.data.cpu().numpy(), query_binary=query_B.data.cpu().numpy(),
                    database_label=db_L, query_label=query_L)
                sio.savemat(os.path.join(args.output_dir, 'epoch-' + str(epoch+1) + '-' +
                                         args.dataset + '-' + str(args.N_bits) + '-' + str(mAP.item()) + ".mat"),
                            {'database_binary': db_B.data.cpu().numpy(), 'query_binary': query_B.data.cpu().numpy(),
                             'database_label': db_L, 'query_label': query_L})

                print("epoch:%d, bit:%d, dataset:%s, best MAP:%.4f" % (epoch + 1, args.N_bits, args.dataset, mAP.item()))
            #log
            # log_line = f"[Epoch {epoch + 1}] C_loss:{C_loss:.4f}, S_loss:{S_loss:.4f}, R_loss:{R_loss:.4f}, mAP:{mAP:.4f}, MAX mAP:{MAX_mAP:.4f}"
            log_line = f"[Epoch {epoch + 1}] C_loss:{C_loss:.4f}, S_loss:{S_loss:.4f}, R_loss:{R_loss:.4f}, mAP:{mAP:.4f}, MAX mAP:{MAX_mAP:.4f}, a:{a:.4f}"
            print(log_line)
            with open(log_path, "a") as f:
                f.write(log_line + "\n")

            net.train()

    if not args.single_gpu:
        dist.destroy_process_group()

def test(args):
    update_from_args(args)  # copy args to config
    device = torch.device("cuda")

    if args.dataset == 'wildfish':
        NB_CLS = 685
        Top_N = 5000
    else:
        raise ValueError("Unsupported dataset")

    config.num_classes = NB_CLS
    config.hash_bit = args.N_bits

    Img_dir = args.data_dir
    Gallery_dir = args.db_txt
    Query_dir = args.query_txt

    if args.encoder == 'ViT':
        model = VisionTransformer(config)
        fc_dim = 768
    else:
        raise ValueError("Only ViT supported in test for now")

    H = Hash_func(fc_dim, args.N_bits, NB_CLS)
    net = nn.Sequential(model, H)
    net = net.to(device)

    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"Loading checkpoint from {args.resume_path}")
        checkpoint = torch.load(args.resume_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("No checkpoint found for testing")

    mAP, query_B, db_B, query_L, db_L = DoRetrieval(
        device, net.eval(), Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args
    )

    print(f"[Test Result] mAP@Top{Top_N}: {mAP:.4f}")

    log_line = f"[Test Result] mAP@Top{Top_N}: {mAP:.4f}"
    with open(args.test_log, "a") as f:
        f.write(log_line + "\n")

    print(f"mAP result saved to: {args.test_log}")

    if args.tsne_vis:
        print("[t-SNE] Extracting features for t-SNE visualization...")

        query_set = ImageList(Img_dir, open(Query_dir).readlines())
        query_loader = torch.utils.data.DataLoader(query_set, batch_size=64, shuffle=True, num_workers=8)

        feats, lbls = extract_features_for_tsne(device, net, query_loader, max_samples=20000)

        unique_classes = np.unique(lbls)
        selected_classes = np.random.choice(unique_classes, size=30, replace=False)
        mask = np.isin(lbls, selected_classes)
        feats = feats[mask]
        lbls = lbls[mask]

        plot_tsne(feats, lbls, title=None, save_path=f"./SNE/tsne_ours_{args.dataset}_{args.N_bits}bit.png")

def extract_features_for_tsne(device, net, dataloader, max_samples=4000):
    net.eval()
    features = []
    labels = []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = (img - mean) / std

            feat = net[0](img)
            features.append(feat.cpu().numpy())
            labels.append(label.cpu().numpy())

            if len(features) * img.size(0) >= max_samples:
                break

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.argmax(labels, axis=1)
    return features, labels

def plot_tsne(features, labels, title, save_path=None):
    tsne = TSNE(n_components=2, perplexity=80, max_iter=3000, random_state=42)
    X_tsne = tsne.fit_transform(features)

    X_tsne = (X_tsne - X_tsne.min(0)) / (X_tsne.max(0) - X_tsne.min(0))
    X_tsne = X_tsne * 2 - 1  # 映射到 [-1, 1]

    num_classes = len(np.unique(labels))
    palette = sns.color_palette("hls", num_classes)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=labels, legend=False, palette=palette, s=30)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DHD', parents=[get_args_parser()])
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='Run mode: train or test')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

#CUDA_VISIBLE_DEVICES=0 python main_DHD.py