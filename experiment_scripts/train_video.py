'''Reproduces Supplement Sec. 7'''

# Enable import from parent package
import sys
import os, time

from numpy import False_
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
import skvideo.datasets

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=2000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=200,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, default='bikes',
               help='Video dataset; one of (cat, bikes)', choices=['cat', 'bikes'])
p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu)')
p.add_argument('--sample_frac', type=float, default=38e-4, # 38e-4 is the default value if split_train is not set.
               help='What fraction of video pixels to sample in each batch (default is all)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--st_split', action='store_true')
p.add_argument('--test_dim', type=int, default=512)
p.add_argument('--speed_test', action='store_true')
p.add_argument('--split_train', action='store_true')
p.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--last_layer_features', type=int, default=-1)
p.add_argument('--lr_decay', type=float, default=1) #0.9995395890030878)

opt = p.parse_args()

if opt.dataset == 'cat':
    video_path = './data/cat_video.mp4'
elif opt.dataset == 'bikes':
    video_path = skvideo.datasets.bikes()

vid_dataset = dataio.Video(video_path)
coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, split_coord=opt.split_train, sample_fraction=opt.sample_frac, batch_size=opt.batch_size) #  frame_sample_fraction=0.608, pixel_sample_fraction=0.00625, 
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=opt.workers, worker_init_fn=dataio.worker_init_fn)

if opt.st_split:
    split_rule = [1,2]
else:
    split_rule = [1,1,1]

# Define the model.
if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh':
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, out_features=vid_dataset.channels, mode='mlp', 
        hidden_features=1024, num_hidden_layers=3, split_mlp=opt.split_mlp, split_rule=split_rule, approx_layers=opt.approx_layers,
        fusion_operator=opt.fusion_operator, last_layer_features=opt.last_layer_features)
elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', in_features=3, out_features=vid_dataset.channels, mode=opt.model_type, 
        hidden_features=1024, num_hidden_layers=3, split_mlp=opt.split_mlp, split_rule=split_rule, approx_layers=opt.approx_layers,
        fusion_operator=opt.fusion_operator, last_layer_features=opt.last_layer_features)
else:
    raise NotImplementedError

# model.module_prefix  = "module."
# model.net.module_prefix = "module."
# model = torch.nn.parallel.DataParallel(model)
# model.load_state_dict(torch.load(opt.checkpoint_path))
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
loss_fn = partial(loss_functions.image_mse, None)
summary_fn = partial(utils.write_video_summary, vid_dataset)
lr_sched = lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, opt.lr_decay)

if not opt.speed_test:
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, lr_sched=lr_sched)

else:
    vid_len = 100
    h = opt.test_dim
    w = opt.test_dim
    if not opt.split_mlp:
        with torch.no_grad():
            pts_in = dataio.get_mgrid([vid_len, h, w], dim=3).reshape(vid_len, -1, 3)
            t0 = time.time()
            for i in range(vid_len):
                model_input = {'coords': pts_in[i].cuda()}
                model_output = model(model_input)
                f"{model_output['model_out'][...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/vid_len}")
    elif not opt.st_split:
        with torch.no_grad():
            x = torch.linspace(-1,1,h).unsqueeze(-1).cuda()
            y = torch.linspace(-1,1,w).unsqueeze(-1).cuda()
            t = torch.linspace(-1,1,vid_len).unsqueeze(-1).cuda()
            t0 = time.time()
            x_feat = model.forward_split_channel(x, 1)
            y_feat = model.forward_split_channel(y, 2)
            t_feat = model.forward_split_channel(t, 0)
            f_feat = x_feat.unsqueeze(1) + y_feat.unsqueeze(0)
            print(t_feat.shape, f_feat.shape)
            for i in range(vid_len):
                model_output = model.forward_split_fusion([f_feat,t_feat[i]])
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/vid_len}")
    else:
        with torch.no_grad():
            xy = pts_in = dataio.get_mgrid([h, w], dim=2).cuda()
            t = torch.linspace(-1,1,vid_len).unsqueeze(-1).cuda()
            t0 = time.time()
            xy_feat = model.forward_split_channel(xy, 1)
            t_feat = model.forward_split_channel(t, 0)
            print(t_feat.shape, xy_feat.shape)
            for i in range(vid_len):
                model_output = model.forward_split_fusion([xy_feat,t_feat[i]])
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/vid_len}")