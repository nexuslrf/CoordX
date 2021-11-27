'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import time
import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--points_per_batch', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=4000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=10,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--act_scale', type=float, default=1)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--fusion_before_act', action='store_true')
p.add_argument('--speed_test', action='store_true')
p.add_argument('--split_train', action='store_true')
p.add_argument('--test_dim', type=int, default=512)
opt = p.parse_args()


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.points_per_batch)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, split_mlp=opt.split_mlp,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if not opt.speed_test:
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary * len(sdf_dataset), epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                clip_grad=True, split_train=opt.split_train)

# # test sdf speed
else:
    test_len = 50
    if not opt.split_mlp:
        with torch.no_grad():
            model_input = {'coords': dataio.get_mgrid(opt.test_dim, 3).cuda()}
            print("test start!")
            t0 = time.time()
            for i in range(test_len):
                model_output = model(model_input)
                f"{model_output['model_out'][...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")
    else:
        with torch.no_grad():
            x = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            y = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            z = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            print("test start!")
            N = opt.test_dim
            t0 = time.time()
            for i in range(test_len):
                x_feat = model.forward_split_channel(x, 0)
                y_feat = model.forward_split_channel(y, 1)
                z_feat = model.forward_split_channel(z, 2)
                sh = list(x_feat.shape)[1:]
                fusion_feat = x_feat.reshape([1,1, N] + sh) + y_feat.reshape([1,N,1] + sh) + z_feat.reshape([N,1,1]+sh)
                model_output = model.forward_split_fusion(x_feat.unsqueeze(1) + y_feat.unsqueeze(0) + z_feat.unsqueeze(0))
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")