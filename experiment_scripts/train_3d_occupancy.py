'''
The migration of 3d occupancy learning tasks from fourier-feat paper to SIREN
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import time, tqdm
import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--points_per_batch', type=int, default=32768)
p.add_argument('--lr', type=float, default=5e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=2000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mesh_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--act_scale', type=float, default=1)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--fusion_before_act', action='store_true')
p.add_argument('--split_train', action='store_true')
p.add_argument('-j', '--workers', default=2, type=int, help='number of data loading workers (default: 4)')
p.add_argument('--recenter', type=str, choices=['fourier', 'siren'], default='fourier')
p.add_argument('--lr_decay', type=float, default=0.9995395890030878) # 0.1 ** (1/5000) = 0.9995395890030878
p.add_argument('--use_atten', action='store_true')
p.add_argument('--last_layer_features', type=int, default=-1)
p.add_argument('--n_hidden_layers', type=int, default=3)
p.add_argument('--prep_cache', action='store_true')
p.add_argument('--pts_cache', type=str, default="")
p.add_argument('--rbatches', type=int, default=1)

p.add_argument('--speed_test', action='store_true')
p.add_argument('--test_dim', type=int, default=512)
p.add_argument('--grid_batch', type=int, default=16)
opt = p.parse_args()

mesh_dataset = dataio.Mesh(opt.mesh_path, pts_per_batch=opt.points_per_batch, num_batches=opt.batch_size, 
        recenter=opt.recenter, split_coord=opt.split_train, pts_cache=opt.pts_cache if not opt.prep_cache else '')
dataloader = DataLoader(mesh_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=opt.workers, worker_init_fn=dataio.worker_init_fn)

if opt.prep_cache:
    total_step = 0
    data = []
    in_label = 'coords_split' if opt.split_train else 'coords'
    with tqdm.tqdm(total=opt.num_epochs) as pbar:
        while total_step < opt.num_epochs:
            for model_input, gt in dataloader:
                data.append({'in': model_input[in_label], 'gt': gt['occupancy']})
                pbar.update(1)
                total_step += 1
    torch.save(data, opt.pts_cache)
    exit()


# Define the model.
if opt.model_type == 'fourier':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp, num_hidden_layers=opt.n_hidden_layers, 
        freq_params=[6., 256//3], include_coord=False, approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, 
        fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten, last_layer_features=opt.last_layer_features)
elif opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp, num_hidden_layers=opt.n_hidden_layers,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, 
        fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten, last_layer_features=opt.last_layer_features)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, split_mlp=opt.split_mlp, num_hidden_layers=opt.n_hidden_layers,
        approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, 
        fusion_before_act=opt.fusion_before_act, use_atten=opt.use_atten, last_layer_features=opt.last_layer_features)
model.cuda()

# Define the loss
loss_fn = loss_functions.occupancy_3d
summary_fn = partial(utils.write_occupancy_summary, mesh_dataset.pts_eval, mesh_dataset, opt.rbatches)
lr_sched = lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, opt.lr_decay)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

if not opt.speed_test:
    t0 = time.time()
    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
                clip_grad=True, split_train=opt.split_train, lr_sched=lr_sched)
    print(f"Training time: {time.time() - t0}")
# # test sdf speed
else:
    test_len = 50
    if not opt.split_mlp:
        with torch.no_grad():
            model_input = dataio.get_mgrid(opt.test_dim, 3).cuda()
            g_batch = (model_input.shape[0] - 1) // opt.grid_batch + 1
            print("test start!")
            t0 = time.time()
            for j in range(test_len):
                model_output = []
                for i in range(opt.grid_batch):
                    input_bin = model_input[i*g_batch:(i+1)*g_batch, :]
                    model_output.append(model.forward({"coords":input_bin})['model_out'])
                model_output = torch.cat(model_output, dim=0)
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")
    else:
        N = opt.test_dim
        gen_sh = lambda i: [1]*i + [N] + [1] * (2-i) + [-1]
        g_batch = (N - 1) // opt.grid_batch + 1
        with torch.no_grad():
            x = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            y = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            z = torch.linspace(-1,1,opt.test_dim).unsqueeze(-1).cuda()
            print("test start!")
            t0 = time.time()
            for j in range(test_len):
                x_feat = model.forward_split_channel(x, 0).reshape(gen_sh(0))
                y_feat = model.forward_split_channel(y, 1).reshape(gen_sh(1))
                z_feat = model.forward_split_channel(z, 2).reshape(gen_sh(2))
                model_output = []
                for i in range(opt.grid_batch):
                    model_output.append(model.forward_split_fusion([x_feat[i*g_batch:(i+1)*g_batch], y_feat, z_feat]))
                model_output = torch.cat(model_output, dim=0)
                f"{model_output[...,0]}"
            t1 = time.time()
            print(f"Time consumed: {(t1-t0)/test_len}")