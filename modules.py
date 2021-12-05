import torch
from torch import nn
from torch.functional import split
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__
    num_input = 1

    def forward(self, input, params=None):
        if params is None:
            weight = self.weight
            bias  = self.bias
        else:
            bias = params.get('bias', None)
            weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2) / self.num_input
        return output


class Sine(nn.Module):
    split_scale = 1

    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * self.split_scale * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False, 
        nonlinearity='relu', weight_init=None, approx_layers=2, fusion_size=1, reduced=False):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        s = 1 if reduced else fusion_size
        in_size = hidden_features * s
        out_size = hidden_features * s

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, out_size), nl
        ))
        for i in range(num_hidden_layers):
            if i+1 == approx_layers:
                out_size = hidden_features if not reduced else hidden_features * fusion_size
            if i == approx_layers:
                out_size = hidden_features
            self.net.append(MetaSequential(
                BatchLinear(in_size, out_size), nl
            ))
            in_size = out_size

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(in_size, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(in_size, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SplitFCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False, 
            nonlinearity='relu', weight_init=None, coord_dim=2, approx_layers=2, split_rule=None, fusion_operator='sum', 
            act_scale=1, fusion_before_act=False, use_atten=False, learn_code=False, last_layer_features=-1, fusion_size=1, reduced=False):
        super().__init__()
        self.first_layer_init = None
        self.coord_dim = coord_dim
        feat_per_channel = in_features // coord_dim
        if split_rule is None:
            self.feat_per_channel = [feat_per_channel] * coord_dim
        else:
            self.feat_per_channel = [feat_per_channel * k for k in split_rule]
        self.split_channels = len(self.feat_per_channel)
        self.approx_layers = approx_layers
        self.num_hidden_layers = num_hidden_layers
        self.module_prefix = ""
        self.fusion_operator = fusion_operator
        self.fusion_before_act = fusion_before_act
        self.out_features = out_features
        self.use_atten = use_atten
        self.learn_code = learn_code
        self.fusion_size = 1
        self.fusion_feat_size = out_features

        if approx_layers != num_hidden_layers + 1:
            last_layer_features = 1
            self.fusion_size = fusion_size
            self.fusion_feat_size = hidden_features
        elif last_layer_features < 0:
            last_layer_features = hidden_features # Channels
        
        last_layer_features = last_layer_features * out_features

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]
        split_scale = act_scale

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        s = 1 if reduced else fusion_size
        self.coord_linears = nn.ModuleList(
            [BatchLinear(feat, hidden_features*s) for feat in self.feat_per_channel]
        )
        self.coord_nl = nl
        self.coord_nl.split_scale = split_scale

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.coord_linears.apply(first_layer_init)
        else:
            self.coord_linears.apply(self.weight_init)
 
        self.net = []
        i = -1
        for i in range(min(approx_layers, num_hidden_layers)-1):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features*s, hidden_features*s), nl
            ))
        i+=1
        self.net.append(MetaSequential(
                BatchLinear(hidden_features*s, hidden_features*fusion_size), nl
            ))
        for j in range(i+1, num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, last_layer_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, last_layer_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
        for i in range(self.approx_layers):
            try:
                self.net[i][0].num_input = self.split_channels
                self.net[i][1].split_scale = split_scale
            except:
                pass
        if use_atten:
            self.atten = BatchLinear(in_features, hidden_features*fusion_size)
            self.atten.apply(self.weight_init)
        if fusion_before_act and nonlinearity.endswith('elu') and \
            self.approx_layers-1 != self.num_hidden_layers + 1:
            
            self.net[self.approx_layers-1][1].inplace = False
        if learn_code:
            self.code = nn.parameter.Parameter(torch.ones(hidden_features*fusion_size))
            

    def forward(self, coords, params=None, pos_codes=None, split_coord=False, ret_feat=False, **kwargs):
        """
        When split_coord=True, the input coords should be a list a tensor for each coord.
        the length of each coord tensor do not need to be the same. But the dimension of each coord tensor
        should be predefined for broadcasting operation.
        """
        # TODO no support for passing params.
        if split_coord:
            hs = [self.forward_channel(coord, i, pos_codes) for i, coord in enumerate(coords)]
            h = self.forward_fusion(hs)
            sh = h.shape
            if ret_feat:
                return (h.reshape(sh[0], -1, sh[-1]), hs)
            else:
                return h.reshape(sh[0], -1, sh[-1])

        hs = torch.split(coords, self.feat_per_channel, dim=-1)
        coord_h = []
        for i, hi in enumerate(hs):
            h = self.coord_linears[i](hi)
            coord_h.append(h)
        hs = torch.stack(coord_h, -2)
        hs = self.coord_nl(hs)

        for i in range(self.approx_layers-1):
            hs = self.net[i](hs)
        
        # layer before fusion
        if not self.fusion_before_act:
            # for simple fusion strategies
            if self.approx_layers > 0:
                hs = self.net[self.approx_layers-1](hs)
            if pos_codes is not None:
                # hs = (hs * pos_codes)
                hs = (hs + pos_codes)
            if self.fusion_operator == 'sum':
                h = hs.sum(-2)
            elif self.fusion_operator == 'prod':
                h = hs.prod(-2)

            if self.use_atten:
                h = h * self.atten(coords)
        else:
            # fusion before activation
            hs = self.net[self.approx_layers-1][0](hs)
            if pos_codes is not None:
                # hs = (hs * pos_codes)
                hs = (hs + pos_codes)
            if self.fusion_operator == 'sum':
                h = hs.sum(-2)
            elif self.fusion_operator == 'prod':
                h = hs.prod(-2)
            
            if self.use_atten:
                h = h * self.atten(coords)
            h = self.net[self.approx_layers-1][1](h)
        if self.learn_code:
            h = h * self.code
        # if self.approx_layers == self.num_hidden_layers + 1:
        #     ### [..., M] --> [..., M//O, O]
        h_sh = h.shape
        h = h.reshape(*h_sh[:-1], self.fusion_feat_size, -1).sum(-1)

        for i in range(self.approx_layers, self.num_hidden_layers+1):
            h = self.net[i](h)
        if ret_feat:
            return (h, hs)
        else:
            return h
    
    def forward_channel(self, coord, channel_id, pos_codes=None):
        h = self.coord_linears[channel_id](coord)
        h = self.coord_nl(h)
        if self.approx_layers > 0:
            for i in range(self.approx_layers-1):
                h = self.net[i](h)
            # layer before fusion
            if not self.fusion_before_act:
                # for simple fusion strategies
                h = self.net[self.approx_layers-1](h)
            else:
                # fusion before activation
                h = self.net[self.approx_layers-1][0](h)
            if pos_codes is not None:
                # h = (h * pos_codes)
                h = (h + pos_codes)
        return h
    
    def forward_fusion(self, hs):
        '''
        When do the fusion, it will expand the list of coord into a grid. 
        In this case, data dimension needs to be predefine. E.g.,
            X: [1,128,1], Y: [64,1,1] --> [64,128,1]
        '''
        # if not isinstance(hs, torch.Tensor):
        h = hs[0]
        for hi in hs[1:]:
            if self.fusion_operator == 'sum':
                h = h + hi
            elif self.fusion_operator == 'prod':
                h = h * hi
        # else:
        #     h = hs
        if self.fusion_before_act:
            h = self.net[self.approx_layers-1][1](h)
        if self.learn_code:
            h = h * self.code
        # if self.approx_layers == self.num_hidden_layers + 1:
        h_sh = h.shape
        if h_sh[-1] > self.fusion_feat_size:
            h = h.reshape(*h_sh[:-1], self.fusion_feat_size, -1).sum(-1)
        for i in range(self.approx_layers, self.num_hidden_layers+1):
            h = self.net[i](h)
        return h


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2, mode='mlp', hidden_features=256, 
        num_hidden_layers=3, split_mlp=False, split_rule=None, pos_enc=False, **kwargs):
        super().__init__()
        print(kwargs)
        self.mode = mode
        self.split_mlp = split_mlp
        self.pos_enc = pos_enc
        self.module_prefix = ""
        coord_dim = in_features
        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True),
                                                       freq_params=kwargs.get('freq_params', None),
                                                       include_coord=kwargs.get('include_coord', True),
                                                       freq_last=split_mlp)
            in_features = self.positional_encoding.out_dim
        elif pos_enc:
            positional_encoding = PosEncodingNeRF(in_features=1,
                                                    sidelength=kwargs.get('sidelength', None),
                                                    fn_samples=kwargs.get('fn_samples', None),
                                                    use_nyquist=kwargs.get('use_nyquist', True),
                                                    freq_last=split_mlp)
            pe_features = positional_encoding.out_dim
            self.pos_encoder = nn.Sequential(
                *[
                    positional_encoding,
                    nn.Linear(pe_features, hidden_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_features, hidden_features)
                ]
            )
            
        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        if not split_mlp:
            self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type,
                               approx_layers=kwargs.get('approx_layers', 2), fusion_size=kwargs.get("fusion_size", 1),
                               reduced=kwargs.get('reduced', False))
        else:
            self.net = SplitFCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type, coord_dim=coord_dim, split_rule=split_rule, 
                               approx_layers=kwargs.get('approx_layers', 2),
                               act_scale=kwargs.get("act_scale", 1),
                               fusion_operator=kwargs.get("fusion_operator", 'prod'),
                               fusion_before_act=kwargs.get("fusion_before_act", False),
                               use_atten=kwargs.get("use_atten", False),
                               learn_code=kwargs.get("learn_code", False),
                               last_layer_features=kwargs.get("last_layer_features", -1),
                               fusion_size=kwargs.get("fusion_size", 1),
                               reduced=kwargs.get('reduced', False)
                               )
        print(self)

    def forward(self, model_input, params=None, ret_feat=False):
        '''
        if coords_split in model_input, then model_input['coords_split'] should be a list of tensors for each coord
        '''
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        if 'coords' in model_input:
            coords_org = model_input['coords'].clone().detach().requires_grad_(True)
            coords = coords_org

            # various input processing methods for different applications
            if self.image_downsampling.downsample:
                coords = self.image_downsampling(coords)
            if self.mode == 'rbf':
                coords = self.rbf_layer(coords)
            elif self.mode == 'nerf':
                coords = self.positional_encoding(coords)

            if self.pos_enc:
                coord_dim = coords.shape[-1]
                pos_codes = self.pos_encoder(coords.unsqueeze(-1))
                pos_codes = pos_codes.reshape(pos_codes.shape[0], -1, coord_dim, pos_codes.shape[-1])
                    # pos_codes = pos_codes.sum(-2, keepdim=True) - pos_codes
                output = self.net(coords, get_subdict(params, self.module_prefix + 'net'), pos_codes=pos_codes)
            else:
                output = self.net(coords, get_subdict(params, self.module_prefix + 'net'))
        if 'coords_split' in model_input:
            coords_org = [coord.clone().detach().requires_grad_(True) for coord in model_input['coords_split']]
            coords = coords_org
            if self.mode == 'nerf':
                coords = [self.positional_encoding(coord, single_channel=True) for coord in coords]
            output = self.net(coords, split_coord=True, ret_feat=ret_feat)
            
        return {'model_in': coords_org, 'model_out': output}


    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}
    
    def forward_split_channel(self, coord, channel_id):
        if not self.split_mlp:
            return None
        # TODO so far only support nerf p.e. for split_coord.
        if self.mode == 'nerf':
            coord = self.positional_encoding(coord, single_channel=True)
        channel_feat = self.net.forward_channel(coord, channel_id)
        return channel_feat
    
    def forward_split_fusion(self, h):
        # h = torch.stack(feats, -1).sum(-1)
        return self.net.forward_fusion(h)


class ImageDownsampling(nn.Module):
    '''Generate samples in u,v plane according to downsampling blur kernel'''

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    '''freq_params: [embedding_scale, embedding_size]
    '''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, freq_last=False, 
        freq_params=None, include_coord=True):

        super().__init__()

        self.in_features = in_features
        self.freq_last = freq_last
        self.include_coord = include_coord

        if freq_params is None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                if fn_samples is None:
                    fn_samples = min(sidelength[0], sidelength[1])
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

            self.out_dim = 2 * in_features * self.num_frequencies
            self.out_dim = self.out_dim + in_features if include_coord else self.out_dim
            self.freq_bands = nn.parameter.Parameter(2**torch.arange(self.num_frequencies) * np.pi, requires_grad=False)
        else:
            self.num_frequencies = freq_params[1]
            self.out_dim = 2 * in_features * self.num_frequencies
            self.out_dim = self.out_dim + in_features if include_coord else self.out_dim

            bval = 2.**torch.linspace(0,freq_params[0], freq_params[1]) - 1
            self.freq_bands = nn.parameter.Parameter(2 * bval * np.pi, requires_grad=False)

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords, single_channel=False):
        if single_channel:
            in_features = coords.shape[-1]
            out_dim = self.out_dim // self.in_features * in_features
        else:
            in_features = self.in_features
            out_dim = self.out_dim

        coords_pos_enc = coords.unsqueeze(-2) * \
            self.freq_bands.reshape([1]*(len(coords.shape)-1) + [-1, 1])
        sin = torch.sin(coords_pos_enc)
        cos = torch.cos(coords_pos_enc)
        coords_pos_enc = torch.cat([sin, cos], -1).reshape(list(coords_pos_enc.shape)[:-2] + [-1])
        if self.include_coord:
            coords_pos_enc = torch.cat([coords, coords_pos_enc], -1)

        if self.freq_last:
            sh = coords_pos_enc.shape[:-1]
            coords_pos_enc = coords_pos_enc.reshape(*sh, -1, in_features).transpose(-1,-2).reshape(*sh, -1)

        return coords_pos_enc


class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

########################
# Initialization methods

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


if __name__=='__main__':
    net = SplitFCBlock(in_features=20, out_features=1, num_hidden_layers=3,
                    hidden_features=64, outermost_linear=True, nonlinearity='relu')
    x = torch.randn(1,16,20)
    y = net(x)
    print(y.shape)