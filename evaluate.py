import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.dice_score import dice_coeff, multiclass_dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize variables to store the results
    total_correct = 0
    total_pixels = 0
    conf_matrix_list = []

    dice_score = 0
    class_iou = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            # Compute other metrics
            mask_pred = mask_pred.argmax(dim=1)
            conf_mat = confusion_matrix(mask_true.cpu().numpy().flatten(), mask_pred.cpu().numpy().flatten())
            conf_matrix_list.append(conf_mat)
            total_correct += (mask_pred == mask_true).sum().item()
            total_pixels += mask_true.numel()

    net.train()

    # Calculate IoU for each class
    conf_matrix = np.mean(conf_matrix_list, axis=0) if conf_matrix_list else np.zeros((net.n_classes, net.n_classes))
    class_iou = []
    for i in range(net.n_classes):
        intersection = conf_matrix[i, i]
        union = np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - intersection
        iou = intersection / union if union != 0 else 0
        class_iou.append(iou)

    # Calculate overall metrics
    accuracy = total_correct / total_pixels
    precision = precision_score(mask_true.cpu().numpy().flatten(), mask_pred.cpu().numpy().flatten(), average=None)
    recall = recall_score(mask_true.cpu().numpy().flatten(), mask_pred.cpu().numpy().flatten(), average=None)
    f1 = f1_score(mask_true.cpu().numpy().flatten(), mask_pred.cpu().numpy().flatten(), average=None)
    # Calculate mean IoU for classes 1 to 3
    mean_iou_macro = np.mean([class_iou[i] for i in range(1, len(class_iou))])

    return {
        'PA': accuracy,
        'CPA': np.mean(conf_matrix.diagonal()) if net.n_classes > 1 else 0,
        'MPA': np.mean(conf_matrix.flatten()),
        'IoU_0': class_iou[0] if net.n_classes > 0 else 0,
        'IoU_1': class_iou[1] if net.n_classes > 1 else 0,
        'IoU_2': class_iou[2] if net.n_classes > 2 else 0,
        'IoU_3': class_iou[3] if net.n_classes > 3 else 0,
        'MIoU': mean_iou_macro,
        'Dice': dice_score / max(num_val_batches, 1),
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
