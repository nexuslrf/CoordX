import enum
import matplotlib.pyplot as plt
import numpy as np
import torch
import dataio
import os
import diff_operators
from torchvision.utils import make_grid, save_image
import skimage.measure
import cv2
import scipy.io.wavfile as wavfile
import cmapy
import loss_functions
from ray_rendering import *

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)

def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = dataio.get_mgrid(512)

    with torch.no_grad():
        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)

def write_occupancy_summary(test_pts, mesh, rbatches, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    with torch.no_grad():

        plot_out = model({'coords': test_pts['pts_plot'].cuda()})
        plot_gt = {'occupancy': test_pts['gt_plot'].cuda()}
        pred = torch.sigmoid(plot_out['model_out']).squeeze(-1)
        psnr = -10.*torch.log10(torch.mean((pred-plot_gt['occupancy'])**2))
        # TODO implement render_rays
        rays = test_pts['render_args_lr'][0]
        rays_H = rays.shape[1]
        rbatch, r_left = rays_H // rbatches, rays_H % rbatches
        rets = []
        for i in range(0, rays_H, rbatch):
            rets.append(render_rays(model, mesh, rays[:,i:i+rbatch], *test_pts['render_args_lr'][1:]))
        if r_left: rets.append(render_rays(model, mesh, rays[:,i+rbatch:] *test_pts['render_args_lr'][1:]))
        depth_map, acc_map = [torch.cat([r[i] for r in rets], 0) for i in range(2)]
        norm_map = make_normals(test_pts['render_args_lr'][0], depth_map) * .5 + .5
        # TODO writer add_image
        writer.add_image(prefix + 'slice_gt', make_grid(plot_gt['occupancy'], scale_each=False, normalize=True),
                     global_step=total_steps)
        writer.add_image(prefix + 'slice_pred', make_grid(pred, scale_each=False, normalize=True),
                     global_step=total_steps)
        # writer.add_image(prefix + 'slice_diff', make_grid((plot_gt['occupancy']-pred).abs(), scale_each=False, normalize=True),
        #              global_step=total_steps)
        writer.add_image(prefix + 'depth_map', make_grid(depth_map, scale_each=False, normalize=True),
                     global_step=total_steps)
        writer.add_image(prefix + 'acc_map', make_grid(acc_map, scale_each=False, normalize=True),
                     global_step=total_steps)
        writer.add_image(prefix + 'norm', make_grid(norm_map.permute(2,0,1), scale_each=False, normalize=True),
                     global_step=total_steps)


        writer.add_scalar('slice_psnr', psnr.cpu().numpy(), total_steps)
        
        for i, (pts, gt) in enumerate(zip(test_pts['pts_metrics'], test_pts['gt_metrics'])):
            pred = model({'coords': pts.cuda()})['model_out'].sigmoid().squeeze(-1)
            gt = gt.cuda()
            val_iou = torch.logical_and(pred > .5, gt > .5).sum() / \
                torch.logical_or(pred > .5, gt > .5).sum()
            writer.add_scalar(f'IoU_{i+1}', val_iou.cpu().numpy(), total_steps)

def write_video_summary(vid_dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    resolution = vid_dataset.shape
    frames = [0, 60, 120, 200]
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    # min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    # min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)


def write_image_summary(image_resolution, model, model_input, gt,
                        model_output, writer, total_steps, prefix='train_'):

    show_grad = model_output['model_out'].shape[-1] == 1
    gt_img = dataio.lin2img(gt['img'], image_resolution)
    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)

    pred_img = dataio.rescale_img((pred_img+1)/2, mode='clamp').permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
    gt_img = dataio.rescale_img((gt_img+1) / 2, mode='clamp').permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    
    writer.add_image(prefix + 'pred_img', torch.from_numpy(pred_img).permute(2, 0, 1), global_step=total_steps)
    
    writer.add_image(prefix + 'gt_img', torch.from_numpy(gt_img).permute(2,0,1), global_step=total_steps)
    
    write_psnr(dataio.lin2img(model_output['model_out'], image_resolution),
               dataio.lin2img(gt['img'], image_resolution), writer, total_steps, prefix+'img_')
               
    if show_grad:
        if 'coords_split' in model_input:
            mgrid = torch.cat(torch.broadcast_tensors(*model_input['coords_split']), -1)
            grid_sh = mgrid.shape
            model_input = {'coords': mgrid.reshape(grid_sh[0], -1, grid_sh[-1])}
            model_output = model(model_input)
        img_gradient = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
        # img_laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
        # img_laplace =  diff_operators.divergence(img_gradient, model_output['model_in'])
        pred_grad = dataio.grads2img(dataio.lin2img(img_gradient, image_resolution)).permute(1,2,0).squeeze().detach().cpu().numpy()
        # pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
        #                          dataio.lin2img(img_laplace), perc=2).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)
        gt_grad = dataio.grads2img(dataio.lin2img(gt['gradients'], image_resolution)).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        # gt_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
            # dataio.lin2img(gt['laplace']), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()), cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)
        writer.add_image(prefix + 'pred_grad', torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=total_steps)
        # writer.add_image(prefix + 'pred_lapl', torch.from_numpy(pred_lapl).permute(2,0,1), global_step=total_steps)
        writer.add_image(prefix + 'gt_grad', torch.from_numpy(gt_grad).permute(2, 0, 1), global_step=total_steps)
        # writer.add_image(prefix + 'gt_lapl', torch.from_numpy(gt_lapl).permute(2, 0, 1), global_step=total_steps)


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)
