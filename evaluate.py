import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.dice_score import dice_coeff, multiclass_dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize variables to store the results
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

            # Calculate confusion matrix on GPU
            conf_mat = torch.zeros(net.n_classes, net.n_classes, device=device)
            for t, p in zip(mask_true.reshape(-1), mask_pred.reshape(-1)):
                conf_mat[t.long(), p.long()] += 1
            conf_matrix_list.append(conf_mat.cpu())  # Move to CPU for averaging

    net.train()

    # Calculate IoU for each class
    conf_matrix = torch.mean(torch.stack(conf_matrix_list, dim=0), dim=0) if conf_matrix_list else torch.zeros(net.n_classes, net.n_classes)
    class_iou = []
    for i in range(net.n_classes):
        intersection = conf_matrix[i, i]
        union = conf_matrix[i].sum() + conf_matrix[:, i].sum() - intersection
        iou = intersection / union if union != 0 else 0
        class_iou.append(iou)

    # Calculate overall metrics
    mean_iou = torch.mean(torch.tensor(class_iou[1:])) if len(class_iou) > 1 else 0

    return {
        'IoU_0': class_iou[0] if net.n_classes > 0 else 0,
        'IoU_1': class_iou[1] if net.n_classes > 1 else 0,
        'IoU_2': class_iou[2] if net.n_classes > 2 else 0,
        'IoU_3': class_iou[3] if net.n_classes > 3 else 0,
        'MIoU': mean_iou.item(),
        'Dice': dice_score / max(num_val_batches, 1),
    }
