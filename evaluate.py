import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.dice_score import dice_coeff, multiclass_dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    class_iou = [0] * net.n_classes  # Initialize IoU scores for each class

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                for i in range(net.n_classes):
                    intersection = (mask_pred[:, i] * mask_true[:, i]).sum().float()
                    union = (mask_pred[:, i].sum() + mask_true[:, i].sum() - intersection).float()
                    if union != 0:
                        class_iou[i] += intersection / union

    net.train()
    dice_score /= num_val_batches
    mean_iou = sum(class_iou[1:4]) / len(class_iou[1:4]) if class_iou[1:4] else 0

    results = {
        'IoU_0': class_iou[0] if class_iou[0] != 0 else 0,
        'IoU_1': class_iou[1] if class_iou[1] != 0 else 0,
        'IoU_2': class_iou[2] if class_iou[2] != 0 else 0,
        'IoU_3': class_iou[3] if class_iou[3] != 0 else 0,
        'MIoU': mean_iou,
        'Dice': dice_score,
    }

    return results

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    hist = np.zeros((net.n_classes, net.n_classes))  # 初始化混淆矩阵
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            # 将预测和真实标签转换为CPU上的numpy数组
            mask_pred = mask_pred.argmax(dim=1).cpu().numpy()
            mask_true = mask_true.argmax(dim=1).cpu().numpy()

            # 更新混淆矩阵
            for lt, lp in zip(mask_true.flatten(), mask_pred.flatten()):
                hist[lt, lp] += 1

    net.train()
    dice_score /= num_val_batches

    # 计算IoU
    class_iou = []
    for i in range(net.n_classes):
        # 计算每个类别的IoU
        intersection = hist[i, i]
        union = hist[i, :].sum() + hist[:, i].sum() - intersection
        iou = intersection / union if union > 0 else 0
        class_iou.append(iou)

    # 计算平均IoU
    mean_iou = np.mean([iou for iou in class_iou[1:] if iou > 0])

    results = {
        'IoU_{}'.format(i): class_iou[i] for i in range(net.n_classes)
    }
    results['MIoU'] = mean_iou
    results['Dice'] = dice_score

    return results