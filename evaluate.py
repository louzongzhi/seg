import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import dice_coeff, multiclass_dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize variables to store the results
    total_dice = 0
    class_iou = [0] * net.n_classes

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)

            total_dice += dice
            mask_pred = mask_pred.argmax(dim=1)
            mask_true = mask_true.view(-1)
            mask_pred = mask_pred.view(-1)

            # Accumulate predictions and true labels for confusion matrix
            for t, p in zip(mask_true, mask_pred):
                class_iou[t.long()] += (t.long() == p.long()).float()

    net.train()

    # Calculate IoU for each class
    mean_iou = sum([iou / (num_val_batches * len(dataloader.dataset)) for iou in class_iou if iou != 0])

    return {
        'IoU_0': class_iou[0] / (num_val_batches * len(dataloader.dataset)) if class_iou[0] != 0 else 0,
        'IoU_1': class_iou[1] / (num_val_batches * len(dataloader.dataset)) if class_iou[1] != 0 else 0,
        'IoU_2': class_iou[2] / (num_val_batches * len(dataloader.dataset)) if class_iou[2] != 0 else 0,
        'IoU_3': class_iou[3] / (num_val_batches * len(dataloader.dataset)) if class_iou[3] != 0 else 0,
        'MIoU': mean_iou,
        'Dice': total_dice / max(num_val_batches, 1),
    }
