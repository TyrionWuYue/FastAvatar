"""
FLAME parameter loss computation with light regularization.
Main supervision comes from rendering loss, this provides minimal regularization.
"""
import torch


def compute_flame_loss(
    pred_dict,
    batch_data,
    flame_param_supervision=False, 
    flame_param_weight=1.0,
    # Supervision weights
    shape_weight=1.0,
    expr_weight=1.0,
    rotation_weight=10.0,
    neck_pose_weight=10.0,
    jaw_pose_weight=10.0,
    eyes_pose_weight=0.1,
    translation_weight=200.0,
    landmark_weight=0.01, 
    img_res=512.0, # Scaling factor for landmarks
    **kwargs
):
    if 'flame_enc_list' in pred_dict and pred_dict['flame_enc_list']:
        flame_params = pred_dict['flame_enc_list'][0]
    elif 'input_flame_outputs' in pred_dict and pred_dict['input_flame_outputs'] is not None:
         # Prioritize input flame outputs as they usually carry the landmarks if available
        flame_params = pred_dict['input_flame_outputs']
    elif 'target_flame_outputs' in pred_dict and pred_dict['target_flame_outputs'] is not None:
         # Fallback to target flame outputs if input not available (e.g. cross reenactment only task?)
        flame_params = pred_dict['target_flame_outputs']
    elif 'flame_outputs' in pred_dict and pred_dict['flame_outputs'] is not None:
        flame_params = pred_dict['flame_outputs']
    else:
        # Fallback to direct dict usage if it looks like params
        flame_params = pred_dict

    # Safety check if we found valid params
    if not isinstance(flame_params, dict) or ('betas' not in flame_params and 'shape' not in flame_params):
        return {'loss_flame': torch.tensor(0.0, device=pred_dict.get('comp_rgb', torch.tensor(0.0)).device if isinstance(pred_dict, dict) else torch.device('cuda'))}

    device = flame_params.get('shape', flame_params.get('betas')).device

    # Dictionary to store all sub-losses
    total_loss = torch.tensor(0.0, device=device)
    loss_dict = {
        'loss_flame': total_loss
    }
    
    # Init all sub-losses to 0.0 for consistent logging
    keys_to_log = ['reg', 'shape', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation', 'landmarks']
    loss_weights = {
        'reg': 1.0, # Placeholder for total reg
        'shape': shape_weight,
        'expr': expr_weight,
        'rotation': rotation_weight,
        'neck_pose': neck_pose_weight,
        'jaw_pose': jaw_pose_weight,
        'eyes_pose': eyes_pose_weight,
        'translation': translation_weight,
        'landmarks': landmark_weight
    }

    for k in keys_to_log:
        loss_dict[f'loss_flame_{k}'] = torch.tensor(0.0, device=device)

    # 1. FLAME Regularization (Prior + GT Supervision)
    loss_reg = torch.tensor(0.0, device=device)
    
    # 1a. Gaussian Prior (L2 to mean face) - "Really Reg"
    # This keeps parameters in a reasonable range even without GT
    shape_p = flame_params.get('shape', flame_params.get('betas'))
    expr_p = flame_params.get('expr')
    
    if shape_p is not None:
        # Standard FLAME prior: L2/weight_decay style
        loss_reg += torch.mean(shape_p**2) * 1e-4 # Small prior for shape
    if expr_p is not None:
        loss_reg += torch.mean(expr_p**2) * 1e-4 # Small prior for expressions

    # 1b. Direct FLAME parameter supervision (when GT available)
    if flame_param_supervision:
        # Build GT flame dict from flat batch_data keys (input_expr, input_rotation, etc.)
        # The dataset outputs flat keys like input_expr, input_rotation, NOT a nested 'flame_param' dict
        gt_flame = {}
        for p_key in ['shape', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']:
            val = batch_data.get(f'input_{p_key}')
            if val is None:
                val = batch_data.get(f'target_{p_key}')
            if val is None and p_key == 'shape':
                val = batch_data.get('input_betas') or batch_data.get('target_betas')
            if val is not None:
                gt_flame[p_key] = val
        param_keys = ['shape', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']
        
        for p_key in param_keys:
             if p_key == 'shape':
                 gt_val = gt_flame.get('shape', gt_flame.get('betas'))
                 pred_val = flame_params.get('shape', flame_params.get('betas'))
             else:
                 gt_val = gt_flame.get(p_key)
                 pred_val = flame_params.get(p_key)
             
             if pred_val is not None and gt_val is not None:
                 if not isinstance(gt_val, torch.Tensor):
                     gt_val = torch.from_numpy(gt_val).to(device)
                 else:
                     gt_val = gt_val.to(device).float()
                 
                 while gt_val.dim() < pred_val.dim():
                     gt_val = gt_val.unsqueeze(1)
                 if gt_val.shape[1] == 1 and pred_val.shape[1] > 1:
                     gt_val = gt_val.expand_as(pred_val)
                 
                 current_weight = loss_weights[p_key]
                 if current_weight > 0:
                    loss_val = torch.mean(torch.abs(pred_val - gt_val))
                    loss_dict[f'loss_flame_{p_key}'] = loss_val
                    loss_reg += current_weight * loss_val

    loss_dict['loss_flame_reg'] = loss_reg
    loss_dict['loss_flame'] += flame_param_weight * loss_reg

    # 2. Landmarks Supervision (Lmk)
    pred_lmk = flame_params.get('landmarks_reproj')
    
    if pred_lmk is not None and 'landmarks' in batch_data and landmark_weight > 0:
        gt_lmk = batch_data['landmarks']
        if not isinstance(gt_lmk, torch.Tensor):
             gt_lmk = torch.tensor(gt_lmk, device=device).float()
        else:
             gt_lmk = gt_lmk.to(device).float()
        
        if gt_lmk.shape[1] > pred_lmk.shape[1]:
             gt_lmk = gt_lmk[:, :pred_lmk.shape[1]]
        
        if pred_lmk.shape == gt_lmk.shape:
             pred_lmk_scaled = pred_lmk * img_res
             if gt_lmk.max() > 2.0:
                 gt_lmk_scaled = gt_lmk
             else:
                 gt_lmk_scaled = gt_lmk * img_res
            
             loss_lmk = torch.mean(torch.abs(pred_lmk_scaled - gt_lmk_scaled))
             loss_dict['loss_flame_landmarks'] = loss_lmk
             loss_dict['loss_flame'] += flame_param_weight * landmark_weight * loss_lmk

    return loss_dict