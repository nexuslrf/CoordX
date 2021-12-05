import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import trimesh
import ray_rendering

# Cavaet! See this: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
# and this: https://discuss.pytorch.org/t/how-to-set-the-same-random-seed-for-all-workers/92253/7
def worker_init_fn(worker_id):                                                          
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    elif dim == 1:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :] = pixel_coords[0, :] / (sidelen[0] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def get_split_shape(shape, dim, n_dim):
    """shape is 2-d array"""
    n_sample = shape[0]
    sample_dim = shape[1]
    sh = [1] * n_dim + [sample_dim]
    sh[dim] = n_sample
    return sh


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True, 
        split_coord=False, samples_per_coord=None):

        super().__init__()

        print("Loading point cloud")
        point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")
        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]
        self.split_coord = split_coord

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

        if self.split_coord:
            self.samples_per_coord = [2 ** int(np.ceil(np.log2(np.cbrt(on_surface_points))))] * 3 \
                if samples_per_coord is None else samples_per_coord

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points if not self.split_coord else np.prod(self.samples_per_coord)
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        if not self.split_coord:
            off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
            coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
            in_dict = {'coords': torch.from_numpy(coords).float()}
        else:
            off_surface_coords = [np.random.uniform(-1, 1, size=get_split_shape((dim, 1), i, 3)) 
                                        for i, dim in enumerate(self.samples_per_coord)]
            coords = on_surface_coords
            coords_split = [torch.from_numpy(coord).float() for coord in off_surface_coords]
            in_dict = {'coords': torch.from_numpy(coords).float(), 'coords_split': coords_split}

        return in_dict, {'sdf': torch.from_numpy(sdf).float(), 'normals': torch.from_numpy(normals).float()}


class Mesh(Dataset):
    def __init__(self, mesh_path, pts_per_batch, keep_aspect_ratio=True, num_batches=1, \
        recenter='fourier', split_coord=False, pts_cache=""):
        super().__init__()
        self.num_batches = num_batches
        self.pts_per_batch = pts_per_batch
        self.split_coord = split_coord

        mesh = trimesh.load(mesh_path)
        """
        Convert a possible scene to a mesh. If conversion occurs, 
            the returned mesh has only vertex and face data.
        """
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in mesh.geometry.values()))
        else:
            assert(isinstance(mesh, trimesh.Trimesh))

        if recenter == 'fourier':
            # Recenter to [0, 1]
            self.pts_range = [0., 1.]
            slice_offset = 0.5
            mesh.vertices -= mesh.vertices.mean(0)
            mesh.vertices /= np.max(np.abs(mesh.vertices))
            mesh.vertices = .5 * (mesh.vertices + 1.)
            self.pts_trans_fn = lambda x: .5 * (x + 1)
        elif recenter == 'siren':
            # SIREN's recentering [-1, 1]
            # it may distort geometry, but makes for high sample efficiency
            self.pts_range = [-1., 1.]
            slice_offset = 0.13
            mesh.vertices -= mesh.vertices.mean(0)
            if keep_aspect_ratio:
                mesh_max = np.amax(mesh.vertices)
                mesh_min = np.amin(mesh.vertices)
            else:
                mesh_max = np.amax(mesh.vertices, axis=0, keepdims=True)
                mesh_min = np.amin(mesh.vertices, axis=0, keepdims=True)
            mesh.vertices = (mesh.vertices - mesh_min) / (mesh_max - mesh_min)
            mesh.vertices -= 0.5
            mesh.vertices *= 2.
            self.pts_trans_fn = lambda x: x * 1.2

        c0, c1 = mesh.vertices.min(0) - 1e-3, mesh.vertices.max(0) + 1e-3

        self.mesh = mesh
        self.corners = (c0, c1)
        
        cache_file = mesh_path.split('.')[0] + f'_{recenter}_test_pts.npy'
        if not os.path.exists(cache_file):
            print('regen pts')
            test_pts = self.make_test_pts()
            np.save(cache_file, test_pts)
        else:
            print('load pts')
            test_pts = np.load(cache_file)

        test_pts = [torch.from_numpy(test).float() for test in test_pts]
        test_gt = [torch.from_numpy(self.gt_fn(test)).float() for test in test_pts]

        N = 256
        x_test = np.linspace(*self.pts_range, N, endpoint=False) * 1.
        x_test = np.stack(np.meshgrid(*([x_test]*2), indexing='ij'), -1)
        pts_plot = np.concatenate([x_test, slice_offset + np.zeros_like(x_test[...,0:1])], -1)
        gt_plot = self.gt_fn(pts_plot)

        # for ray_rendering
        R = 2.
        c2w = ray_rendering.pose_spherical(90. + 10 + 45, -30., R)
        N_samples = 64
        N_samples_2 = 64
        H = 180
        W = H
        focal = H * .9
        render_args_lr = [
            ray_rendering.get_rays(H, W, focal, c2w[:3,:4]), self.corners, 
                R-1, R+1, N_samples, N_samples_2, True, self.pts_trans_fn]

        if self.split_coord:
            vol = (self.corners[1] - self.corners[0]).prod()
            gamma = ((self.pts_per_batch / vol) ** (1/3)).item()
            self.pts_per_axis = (self.corners[1] - self.corners[0]) * gamma
            self.mu = 0.4 # (1+x)^2(1-x)=1

        self.pts_eval = {
            'pts_metrics': test_pts,
            'gt_metrics': test_gt,
            'pts_plot': torch.from_numpy(pts_plot).float(),
            'gt_plot': torch.from_numpy(gt_plot).float(),
            'render_args_lr': render_args_lr
        }

        self.pts_cache = None
        if pts_cache:
            self.pts_cache = torch.load(pts_cache)
            self.num_batches = len(self.pts_cache)
            
    def make_test_pts(self, test_size=2**18):
        c0, c1 = self.corners
        test_easy = np.random.uniform(size=[test_size, 3]) * (c1-c0) + c0
        batch_pts, batch_normals = self.get_normal_batch(test_size)
        test_hard = batch_pts + np.random.normal(size=[test_size,3]) * .01
        return test_easy, test_hard
    
    def get_normal_batch(self, bsize):
        def uniform_bary(u):
            su0 = np.sqrt(u[..., 0])
            b0 = 1. - su0
            b1 = u[..., 1] * su0
            return np.stack([b0, b1, 1. - b0 - b1], -1)
        batch_face_inds = np.array(np.random.randint(0, self.mesh.faces.shape[0], [bsize]))
        batch_barys = np.array(uniform_bary(np.random.uniform(size=[bsize, 2])))
        batch_faces = self.mesh.faces[batch_face_inds]
        batch_normals = self.mesh.face_normals[batch_face_inds]
        batch_pts = np.sum(self.mesh.vertices[batch_faces] * batch_barys[...,None], 1)

        return batch_pts, batch_normals
    
    def gt_fn(self, pts, split_coord=False):
        if split_coord:
            x, y, z = np.meshgrid(np.squeeze(pts[0]), np.squeeze(pts[1]), np.squeeze(pts[2]), indexing='ij')
            final = np.stack([x,y,z],axis=-1).reshape([-1,3])
            return self.mesh.ray.contains_points(final).reshape(final.shape[:-1])
        else:
            return self.mesh.ray.contains_points(pts.reshape([-1,3])).reshape(pts.shape[:-1])

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.pts_cache is not None:
            in_label = 'coords_split' if self.split_coord else 'coords'
            return {in_label: self.pts_cache[idx]['in']}, {'occupancy': self.pts_cache[idx]['gt']}
        elif self.split_coord:
            rand_scale = (torch.rand(3) * 2 - 1) * self.mu + 1
            r = 1/rand_scale[0]
            rb = ((r/(1+self.mu)).clamp_min(1-self.mu), (r/(1-self.mu)).clamp_max(1+self.mu))
            rand_scale[1] = torch.rand(1) * (rb[1]-rb[0]) + rb[0]
            rand_scale[2] = 1 / rand_scale[:2].prod()
            pts = [(torch.rand(round((self.pts_per_axis[i]*rand_scale[i]).item()), 1) \
                    * (self.corners[1][i]-self.corners[0][i]) + self.corners[0][i]) for i in range(3)]
            gt = self.gt_fn(pts, split_coord=True)[...,None]
            return {'coords_split': [pts[0].view(-1, 1, 1, 1), pts[1].view(1, -1, 1, 1), pts[2].view(1, 1, -1, 1)]},\
                {'occupancy': torch.from_numpy(gt).float()}
        else:
            pts = np.random.uniform(size=[self.pts_per_batch, 3]) * \
                (self.corners[1]-self.corners[0]) + self.corners[0]
            gt = self.gt_fn(pts)[...,None]
            return {'coords': torch.from_numpy(pts).float()}, {'occupancy': torch.from_numpy(gt).float()}

class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid


class Camera(Dataset):
    def __init__(self, img_src=None, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        if img_src is None:
            img_src = skimage.data.camera()
        self.img = Image.fromarray(img_src)
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None, split_coord=False):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.split_coord = split_coord
        if not split_coord:
            self.mgrid = get_mgrid(sidelength)
        else:
            n_dim = 2
            self.mgrid = [get_mgrid(sidelength[i], dim=1).reshape(get_split_shape((sidelength[i],1),i,n_dim))
                            for i in range(n_dim)]

        self.skip = False
        self.in_dict, self.gt_dict = None, None
        if len(self.dataset) == 1:
            self.in_dict, self.gt_dict = self.__getitem__(0)
            self.skip = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.skip:
            return self.in_dict, self.gt_dict
        img = self.transform(self.dataset[idx])

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).transpose(1,2,0)
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).transpose(1,2,0)
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).transpose(1,2,0)
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).transpose(1,2,0)
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).transpose(1,2,0)
            laplace = scipy.ndimage.laplace(img.numpy()).transpose(1,2,0)

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        coords_key = 'coords' if not self.split_coord else 'coords_split'
        in_dict = {'idx': idx, coords_key: self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, gradx.shape[-1]),
                                   torch.from_numpy(grady).reshape(-1, gradx.shape[-1])),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, laplace.shape[-1])})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


class Implicit2DWrapperSingleChannel(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset

        self.mgrid_x = get_mgrid(sidelength[0], dim=1)
        self.mgrid_y = get_mgrid(sidelength[1], dim=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])
        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords_x': self.mgrid_x, 'coords_y': self.mgrid_y}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1., batch_size=1, split_coord=False, frame_sample_fraction=1., pixel_sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.batch_size = batch_size
        self.split_coord = split_coord

        data = (torch.from_numpy(self.dataset[0]) - 0.5) / 0.5
        self.data = data.view(-1, self.dataset.channels)
        self.sample_fraction = sample_fraction

        if split_coord:
            self.mgrid_xy = get_mgrid((sidelength[1], sidelength[2]), dim=2).unsqueeze(0)
            self.mgrid_z = get_mgrid(sidelength[0], dim=1).unsqueeze(1)
            self.n_z = sidelength[0]
            self.n_xy = sidelength[1] * sidelength[2]
            self.N_samples = self.n_z * self.n_xy * self.sample_fraction
        else:
            self.mgrid = get_mgrid(sidelength, dim=3)
            self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

    def __len__(self):
        return len(self.dataset) * self.batch_size

    def __getitem__(self, idx):
        if self.split_coord:
            z_num = np.random.randint(round(self.n_z*0.1), round(self.n_z*0.9))
            coord_idx = torch.randperm(self.n_z)[:z_num]
            coord_idy = torch.randint(0, self.n_xy, (round(self.N_samples/z_num),))
            coord_id = (coord_idx[:,None] * self.n_xy + coord_idy[None, :]).reshape(-1)
            data = self.data[coord_id, :]
            coords_xy = self.mgrid_xy[:, coord_idy, :]
            coords_z = self.mgrid_z[coord_idx, :, :]
            in_dict = {'idx': idx, 'coords_split': [coords_z, coords_xy]}
            gt_dict = {'img': data}
        else:
            if self.sample_fraction < 1.:
                coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
                data = self.data[coord_idx, :]
                coords = self.mgrid[coord_idx, :]
            else:
                coords = self.mgrid
                data = self.data
            in_dict = {'idx': idx, 'coords': coords}
            gt_dict = {'img': data}

        return in_dict, gt_dict
