import torch
import torch.nn.functional as F

import diff_operators
import modules


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}

def image_svd(model_output, gt):
    img_pred, feats = model_output['model_out']
    feats = [feat.squeeze() for feat in feats]
    return {
        'img_loss': ((img_pred - gt['img']) ** 2).mean(),
        'orth_reg_x': torch.square((feats[0] @ feats[0].T) - torch.eye(feats[0].shape[0]).cuda()).mean() * 0.1,
        'orth_reg_y': torch.square((feats[1] @ feats[1].T) - torch.eye(feats[1].shape[0]).cuda()).mean() * 0.1,
        }

def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

def occupancy_3d(model_output, gt):
    x = model_output['model_out']
    z = gt['occupancy']
    loss = torch.mean(torch.relu(x) - x * z + torch.log(1 + torch.exp(-torch.abs(x))))
    return {'entropy': loss}