'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)
p.add_argument('--split_mlp', action='store_true')
p.add_argument('--approx_layers', type=int, default=2)
p.add_argument('--act_scale', type=float, default=1)
p.add_argument('--fusion_operator', type=str, choices=['sum', 'prod'], default='prod')
p.add_argument('--fusion_before_act', action='store_true')
opt = p.parse_args()


class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.model_type == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, split_mlp=opt.split_mlp, 
            approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
        else:
            self.model = modules.SingleBVPNet(type=opt.model_type, in_features=3, split_mlp=opt.split_mlp, 
            approx_layers=opt.approx_layers, act_scale=opt.act_scale, fusion_operator=opt.fusion_operator, fusion_before_act=opt.fusion_before_act)
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


sdf_decoder = SDFDecoder()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)
print("creating mesh!")
sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, 'test'), N=opt.resolution)
