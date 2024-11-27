# 导入标准库
import argparse  # 命令行参数解析
import logging  # 日志记录
import os  # 操作系统接口

# 导入PyTorch相关库
import torch  # PyTorch库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
from torchvision import transforms  # 图像预处理
from torchvision.transforms import functional as TF  # 图像预处理函数

# 导入路径操作库
from pathlib import Path  # 路径操作

# 导入PyTorch数据加载和优化器
from torch import optim  # 优化器
from torch.utils.data import DataLoader  # 数据加载器

# 导入进度条显示库
from tqdm import tqdm  # 进度条

# 导入Wandb实验跟踪库
import wandb  # Wandb实验跟踪

# 导入自定义模块
from evaluate import evaluate  # 模型评估
from models import load_model  # 加载模型
from utils.data_loading import BasicDataset, CarvanaDataset  # 数据加载
from utils.dice_score import dice_loss  # Dice损失函数

# 设置数据路径
dir_img_train = Path('./NEU_Seg/images/training/')  # 训练集图像路径
dir_mask_train = Path('./NEU_Seg/annotations/training/')  # 训练集掩码路径
dir_img_test = Path('./NEU_Seg/images/test/')  # 测试集图像路径
dir_mask_test = Path('./NEU_Seg/annotations/test/')  # 测试集掩码路径

# 设置模型检查点保存路径
dir_checkpoint = Path('./checkpoints/')  # 检查点根目录
dir_checkpoint_history = dir_checkpoint / 'history'  # 历史检查点保存路径
dir_checkpoint_best = dir_checkpoint / 'best'  # 最佳检查点保存路径


def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    dir_img_train=dir_img_train,
    dir_mask_train=dir_mask_train,
    dir_img_test=dir_img_test,
    dir_mask_test=dir_mask_test,
):
    # 1. Create dataset for training
    try:
        train_dataset = CarvanaDataset(dir_img_train, dir_mask_train, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        train_dataset = BasicDataset(dir_img_train, dir_mask_train, img_scale)

    # 2. Create dataset for validation
    try:
        val_dataset = CarvanaDataset(dir_img_test, dir_mask_test, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        val_dataset = BasicDataset(dir_img_test, dir_mask_test, img_scale)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(
        f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        '''
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    best_score = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (len(train_dataset) // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), str(dir_checkpoint_best / 'model.pth'))
                logging.info(f'Best checkpoint saved!')
            else:
                torch.save(model.state_dict(), str(dir_checkpoint_history / f'checkpoint_epoch{epoch}.pth'))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = load_model("UNet")
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            'Detected OutOfMemoryError!'
            'Enabling checkpointing to reduce memory usage, but this slows down training.'
            'Consider enabling AMP (--amp) for fast and memory efficient training'
        )
        torch.cuda.empty_cache()
