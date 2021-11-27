'''
Ray rendering related helper function.
Copied from Fourier-Feat's JAX code.
'''
import torch
import numpy as np
from tqdm import tqdm

def get_rays(H, W, focal, c2w):
    j, i = torch.meshgrid(torch.arange(H), torch.arange(W))
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # rays_d = torch.sum(dirs[..., torch.newaxis, :] * c2w[:3,:3], -1)
    rays_d = torch.sum(torch.unsqueeze(dirs, -2) * c2w[:3,:3], -1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return torch.stack([rays_o, rays_d], 0)

trans_t = lambda t : torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=torch.float32)

rot_phi = lambda phi : torch.tensor([
    [1,0,0,0],
    [0,torch.cos(phi),-torch.sin(phi),0],
    [0,torch.sin(phi), torch.cos(phi),0],
    [0,0,0,1],
], dtype=torch.float32)

rot_theta = lambda th : torch.tensor([
    [torch.cos(th),0,-torch.sin(th),0],
    [0,1,0,0],
    [torch.sin(th),0, torch.cos(th),0],
    [0,0,0,1],
], dtype=torch.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(torch.tensor(phi/180.*np.pi)) @ c2w
    c2w = rot_theta(torch.tensor(theta/180.*np.pi)) @ c2w
    return c2w

def vol_render(model, mesh_dataset, rbatch, rays, render_args):
    H = rays.shape[1]
    rets = []
    for i in tqdm(range(0, H, rbatch)):
        rets.append(
            render_rays(model, mesh_dataset, rays[:,i:i+rbatch], *render_args)
        )
    depth_map, acc_map = [torch.cat([r[i] for r in rets], 0) for i in range(2)]
    return depth_map, acc_map


def render_rays(model, mesh, rays, corners, near, far, 
        N_samples, N_samples_2, clip, pts_trans_fn, fine_pass=True):

    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    c0, c1 = (torch.as_tensor(c).cuda() for c in corners)
    
    th = .5
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples).cuda()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = pts_trans_fn(pts)
    # Run network
    with torch.no_grad():
        alpha = model({'coords': pts})['model_out'].sigmoid().squeeze(-1)
    if clip:
        mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1))
        alpha = torch.where(mask, torch.FloatTensor([0.]).cuda(), alpha)

    alpha = (alpha > th).float()

    trans = 1.-alpha + 1e-10
    trans = torch.cat([torch.ones_like(trans[...,:1]).cuda(), trans[...,:-1]], -1)  
    weights = alpha * torch.cumprod(trans, -1)
    
    depth_map = torch.sum(weights * z_vals, -1) 
    acc_map = torch.sum(weights, -1)

    if fine_pass:
        # Second pass to refine isosurface
        z_vals = torch.linspace(-1., 1., N_samples_2).cuda() * .01 + depth_map[...,None]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        pts = pts_trans_fn(pts)
        # Run network
        with torch.no_grad():
            alpha = model({'coords': pts.cuda()})['model_out'].sigmoid().squeeze(-1)
        if clip:
            # alpha = np.where(np.any(np.abs(pts) > 1, -1), 0., alpha)
            mask = torch.logical_or(torch.any(pts < c0, -1), torch.any(pts > c1, -1))
            alpha = torch.where(mask, torch.FloatTensor([0.]).cuda(), alpha)

        alpha = (alpha > th).float()

        trans = 1.-alpha + 1e-10
        trans = torch.cat([torch.ones_like(trans[...,:1]).cuda(), trans[...,:-1]], -1)  
        weights = alpha * torch.cumprod(trans, -1)
        
        depth_map = torch.sum(weights * z_vals, -1) 
        acc_map = torch.sum(weights, -1)

    return depth_map, acc_map

def make_normals(rays, depth_map):
    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    pts = rays_o + rays_d * depth_map[...,None]
    dx = pts - torch.roll(pts, -1, 0)
    dy = pts - torch.roll(pts, -1, 1)
    normal_map = torch.cross(dx, dy)
    normal_map = normal_map / torch.norm(normal_map, dim=-1, keepdim=True).clamp(1e-5)
    return normal_map

def render_mesh_normals(mesh, rays):
    origins, dirs = rays.reshape([2,-1,3])
    origins = origins * .5 + .5
    dirs = dirs * .5
    z = mesh.ray.intersects_first(origins, dirs)
    pic = np.zeros([origins.shape[0],3]) 
    pic[z!=-1] = mesh.face_normals[z[z!=-1]]
    pic = np.reshape(pic, rays.shape[1:])
    return pic

def get_bins(pts, resolution, vmin, vmax, eps=1e-5):
    lbnd = pts.min().clamp_min(vmin)
    ubnd = pts.max().clamp_max(vmax)
    bins = torch.linspace(lbnd, ubnd+eps, resolution+1) # including two end point.
    dstep = (ubnd + eps - lbnd) / resolution
    return bins, dstep, lbnd, ubnd

def get_pts_pred(model, pts_idx, feats, split=True, resolution=[128,128,128]):
    lbnd_idx = pts_idx.long()
    if split:
        ubnd_idx = lbnd_idx + 1
        r = pts_idx - lbnd_idx
        # indexing feats
        lbnd_feats = torch.stack([feats[i][lbnd_idx[...,i]] for i in range(3)])
        ubnd_feats = torch.stack([feats[i][ubnd_idx[...,i]] for i in range(3)])
        # interpolation
        pts_feats = (lbnd_feats * (1 - r).T[...,None] + r.T[...,None] * ubnd_feats)
        with torch.no_grad():
            out = model.forward_split_fusion(pts_feats)
    else:
        offsets = torch.tensor([(resolution[2] + 1) * (resolution[1]+1), resolution[2]+1, 1]).cuda()
        r = pts_idx[..., 0] - lbnd_idx[..., 0]
        # interpolation along x
        # y = 0, z = 0
        ubnd_idx = lbnd_idx + torch.tensor([1, 0, 0]).cuda()
        lbnd_idx_t = lbnd_idx + torch.tensor([0, 0, 0]).cuda() # for demonstration purpose...
        lbnd_idx_mixed = (lbnd_idx_t * offsets).sum(-1)
        lbnd_feats = feats[lbnd_idx_mixed]
        ubnd_idx_mixed = (ubnd_idx * offsets).sum(-1)
        ubnd_feats = feats[ubnd_idx_mixed]
        pts_feats_00 = (lbnd_feats * (1 - r).T[...,None] + r.T[...,None] * ubnd_feats)
        # y = 0, z = 1
        ubnd_idx = lbnd_idx + torch.tensor([1, 0, 1]).cuda()
        lbnd_idx_t = lbnd_idx + torch.tensor([0, 0, 1]).cuda()
        lbnd_idx_mixed = (lbnd_idx_t * offsets).sum(-1)
        lbnd_feats = feats[lbnd_idx_mixed]
        ubnd_idx_mixed = (ubnd_idx * offsets).sum(-1)
        ubnd_feats = feats[ubnd_idx_mixed]
        pts_feats_01 = (lbnd_feats * (1 - r).T[...,None] + r.T[...,None] * ubnd_feats)
        # y = 1, z = 0
        ubnd_idx = lbnd_idx + torch.tensor([1, 1, 0]).cuda()
        lbnd_idx_t = lbnd_idx + torch.tensor([0, 1, 0]).cuda()
        lbnd_idx_mixed = (lbnd_idx_t * offsets).sum(-1)
        lbnd_feats = feats[lbnd_idx_mixed]
        ubnd_idx_mixed = (ubnd_idx * offsets).sum(-1)
        ubnd_feats = feats[ubnd_idx_mixed]
        pts_feats_10 = (lbnd_feats * (1 - r).T[...,None] + r.T[...,None] * ubnd_feats)
        # y = 1, z = 1
        ubnd_idx = lbnd_idx + torch.tensor([1, 1, 1]).cuda()
        lbnd_idx_t = lbnd_idx + torch.tensor([0, 1, 1]).cuda()
        lbnd_idx_mixed = (lbnd_idx_t * offsets).sum(-1)
        lbnd_feats = feats[lbnd_idx_mixed]
        ubnd_idx_mixed = (ubnd_idx * offsets).sum(-1)
        ubnd_feats = feats[ubnd_idx_mixed]
        pts_feats_11 = (lbnd_feats * (1 - r).T[...,None] + r.T[...,None] * ubnd_feats)

        # interpolation along y
        r = pts_idx[..., 1] - lbnd_idx[..., 1]
        # z = 0
        pts_feats_0 = (pts_feats_00 * (1 - r).T[...,None] + r.T[...,None] * pts_feats_10)
        # z = 1
        pts_feats_1 = (pts_feats_01 * (1 - r).T[...,None] + r.T[...,None] * pts_feats_11)

        # interpolation along z
        r = pts_idx[..., 2] - lbnd_idx[..., 2]
        pts_feats = (pts_feats_0 * (1 - r).T[...,None] + r.T[...,None] * pts_feats_1)
        out = pts_feats
    return out

# vol_render function with splitting acceleration
def vol_render_split(model, mesh, rbatch, rays, render_args, fine_pass=False, resolution=256, precompute=None):

    corners, near, far, N_samples, N_samples_2, clip, pts_trans_fn = render_args[:7]
    if len(render_args) > 7: fine_pass = render_args[7]

    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    c0, c1 = corners
    th = .5
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples).cuda()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = pts_trans_fn(pts)

    if precompute is None:
        bins, dsteps = [], []
        vmins, vmaxs = [], []
        try:
            z = resolution[1]
        except:
            resolution = [resolution] * 3
        for i in range(3):
            bin,dstep,vmin,vmax = get_bins(pts[...,i], resolution[i], c0[i], c1[i])
            dsteps.append(dstep)
            vmins.append(vmin); vmaxs.append(vmax)
            bins.append(bin)
        dsteps = torch.stack(dsteps)
        vmins, vmaxs = torch.stack(vmins), torch.stack(vmaxs)
        with torch.no_grad():
            feats = [model.forward_split_channel(b[...,None].cuda(), i) for i,b in enumerate(bins)]
    else:
        vmins = precompute['vmins']
        vmaxs = precompute['vmaxs']
        feats = precompute['feats']
        dsteps = precompute['dsteps']

    pts_idx = (pts - vmins) / dsteps
    # pts within the training bounds
    mask = ~torch.logical_or(torch.any(pts < vmins, -1), torch.any(pts > vmaxs, -1))
    pts_infer_idx = pts_idx[mask] # [P_nz, 3]
    num_pts_infer = pts_infer_idx.shape[0]

    # TODO consider special property of `y_w` later.

    rets = []
    for i in tqdm(range(0, num_pts_infer, rbatch)):
        pts_idx_chunk = pts_infer_idx[i:i+rbatch, :]
        rets.append(get_pts_pred(model, pts_idx_chunk,feats))
    alpha = torch.cat(rets, 0).sigmoid().squeeze()

    ############
    # A faster implementation
    alpha = (alpha > th)
    m_idx = mask.reshape(-1, N_samples).nonzero(as_tuple=True)
    a_nz_idx = alpha.nonzero(as_tuple=True)[0]
    h_id = m_idx[0][a_nz_idx] + 1 # to avoid the first idx is 0
    w_id = m_idx[1][a_nz_idx]
    n_id = (h_id - torch.cat([torch.IntTensor([0]).cuda(), h_id[:-1]])).nonzero(as_tuple=True)[0]
    r_id = h_id[n_id] - 1
    w_id = w_id[n_id]
    depth = z_vals[w_id]
    depth_map = torch.zeros(rays_o.shape[:-1]).cuda().reshape(-1)\
        .scatter(0,r_id,depth).reshape(rays_o.shape[:-1])
    acc_map = (depth_map > 0).float()
    ############
    
    # second level sampling
    if fine_pass:
        z_vals = torch.linspace(-1., 1., N_samples_2).cuda() * .01 + depth[...,None]
        pts = rays_o.reshape(-1,3)[r_id][...,None,:] + \
            rays_d.reshape(-1,3)[r_id][...,None,:] * z_vals[...,:,None]
        pts = pts_trans_fn(pts) # [R, N, 3]
        pts_idx = ((pts - vmins) / dsteps) #.reshape(-1,3) #.clamp(0,resolution+1-1e-10)
        
        mask = ~torch.logical_or(torch.any(pts < vmins, -1), torch.any(pts > vmaxs, -1))
        pts_infer_idx = pts_idx[mask] # [P_nz, 3]

        num_pts_infer = pts_infer_idx.shape[0]

        rets = []
        for i in tqdm(range(0, num_pts_infer, rbatch)):
            pts_idx_chunk = pts_infer_idx[i:i+rbatch, :]
            rets.append(get_pts_pred(model, pts_idx_chunk,feats))
        alpha = torch.cat(rets, 0).sigmoid().squeeze()


        alpha = (alpha > th)
        m_idx = mask.reshape(-1, N_samples).nonzero(as_tuple=True)
        a_nz_idx = alpha.nonzero(as_tuple=True)[0]
        h_id = m_idx[0][a_nz_idx] + 1 # to avoid the first idx is 0
        w_id = m_idx[1][a_nz_idx]
        n_id = (h_id - torch.cat([torch.IntTensor([0]).cuda(), h_id[:-1]])).nonzero(as_tuple=True)[0]
        # r_id = h_id[n_id] - 1
        w_id = w_id[n_id]
        depth = z_vals[torch.arange(z_vals.shape[0]), w_id] #z_vals[w_id]
        depth_map = torch.zeros(rays_o.shape[:-1]).cuda().reshape(-1)\
            .scatter(0,r_id,depth).reshape(rays_o.shape[:-1])
        acc_map = (depth_map > 0).float()

    return depth_map, acc_map

def vol_render_nosplit(model, mesh, rbatch, rays, render_args, fine_pass=False, resolution=256, grid_batch=16, precompute=None):

    corners, near, far, N_samples, N_samples_2, clip, pts_trans_fn = render_args[:7]
    try:
        z = resolution[1]
    except:
        resolution = [resolution] * 3
    if len(render_args) > 7: fine_pass = render_args[7]

    rays_o, rays_d = rays[0].cuda(), rays[1].cuda()
    # c0, c1 = (torch.as_tensor(c).cuda() for c in corners)
    c0, c1 = corners
    th = .5
    
    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples).cuda()
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    pts = pts_trans_fn(pts)

    if precompute is None:
        bins, dsteps = [], []
        vmins, vmaxs = [], []
        for i in range(3):
            bin,dstep,vmin,vmax = get_bins(pts[...,i], resolution[i], c0[i], c1[i])
            dsteps.append(dstep)
            vmins.append(vmin); vmaxs.append(vmax)
            bins.append(bin)
        dsteps = torch.stack(dsteps)
        vmins, vmaxs = torch.stack(vmins), torch.stack(vmaxs)
        bins = torch.stack(torch.meshgrid(bins[0], bins[1], bins[2]), axis=-1).view(-1, 3)
        feats = []
        g_batch = bins.shape[0] // grid_batch + 1
        for i in range(grid_batch):
            with torch.no_grad():
                input_bin = bins[i*g_batch:(i+1)*g_batch, :].cuda()
                feats.append(model.forward({"coords":input_bin})['model_out'])
        feats = torch.cat(feats, dim=0).cuda()
    else:
        vmins = precompute['vmins']
        vmaxs = precompute['vmaxs']
        dsteps = precompute['dsteps']
        feats = precompute['feats']

    pts_idx = (pts - vmins) / dsteps
    mask = ~torch.logical_or(torch.any(pts < vmins, -1), torch.any(pts > vmaxs, -1))
    pts_infer_idx = pts_idx[mask] # [P_nz, 3]
    num_pts_infer = pts_infer_idx.shape[0]

    rets = []
    for i in tqdm(range(0, num_pts_infer, rbatch)):
        pts_idx_chunk = pts_infer_idx[i:i+rbatch, :]
        rets.append(get_pts_pred(model, pts_idx_chunk,feats, split=False, resolution=resolution))
    alpha = torch.cat(rets, 0).sigmoid().squeeze()

    alpha = (alpha > th)
    m_idx = mask.reshape(-1, N_samples).nonzero(as_tuple=True)
    a_nz_idx = alpha.nonzero(as_tuple=True)[0]
    h_id = m_idx[0][a_nz_idx] + 1 # to avoid the first idx is 0
    w_id = m_idx[1][a_nz_idx]
    n_id = (h_id - torch.cat([torch.IntTensor([0]).cuda(), h_id[:-1]])).nonzero(as_tuple=True)[0]
    r_id = h_id[n_id] - 1
    w_id = w_id[n_id]
    depth = z_vals[w_id]
    depth_map = torch.zeros(rays_o.shape[:-1]).cuda().reshape(-1)\
        .scatter(0,r_id,depth).reshape(rays_o.shape[:-1])
    acc_map = (depth_map > 0).float()

    if fine_pass:
        z_vals = torch.linspace(-1., 1., N_samples_2).cuda() * .01 + depth[...,None]
        pts = rays_o.reshape(-1,3)[r_id][...,None,:] + \
            rays_d.reshape(-1,3)[r_id][...,None,:] * z_vals[...,:,None]
        pts = pts_trans_fn(pts) # [R, N, 3]
        pts_idx = ((pts - vmins) / dsteps) #.reshape(-1,3) #.clamp(0,resolution+1-1e-10)
        
        mask = ~torch.logical_or(torch.any(pts < vmins, -1), torch.any(pts > vmaxs, -1))
        pts_infer_idx = pts_idx[mask] # [P_nz, 3]
        num_pts_infer = pts_infer_idx.shape[0]

        rets = []
        for i in tqdm(range(0, num_pts_infer, rbatch)):
            pts_idx_chunk = pts_infer_idx[i:i+rbatch, :]
            rets.append(get_pts_pred(model, pts_idx_chunk,feats, split=False, resolution=resolution))
        alpha = torch.cat(rets, 0).sigmoid().squeeze()

        alpha = (alpha > th)
        m_idx = mask.reshape(-1, N_samples).nonzero(as_tuple=True)
        a_nz_idx = alpha.nonzero(as_tuple=True)[0]
        h_id = m_idx[0][a_nz_idx] + 1 # to avoid the first idx is 0
        w_id = m_idx[1][a_nz_idx]
        n_id = (h_id - torch.cat([torch.IntTensor([0]).cuda(), h_id[:-1]])).nonzero(as_tuple=True)[0]
        w_id = w_id[n_id]
        depth = z_vals[torch.arange(z_vals.shape[0]), w_id] #z_vals[w_id]
        depth_map = torch.zeros(rays_o.shape[:-1]).cuda().reshape(-1)\
            .scatter(0,r_id,depth).reshape(rays_o.shape[:-1])
        acc_map = (depth_map > 0).float()

    return depth_map, acc_map
