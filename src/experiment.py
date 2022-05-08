import pytorch_lightning as pl
import numpy as np
import torch
from PIL import Image
from model import EstimatorModel
from data import DepthEstimatorDataset, collate_fn

class EstimatorExperiment(pl.LightningModule):
    def __init__(self, data_dir, learning_rate, batch_size, num_workers, test_steps, num_images, log_loss_rate):
        super().__init__()

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_steps = test_steps
        self.num_images = num_images
        self.log_loss_rate = log_loss_rate

        self.model = EstimatorModel()
        self.depth_loss = torch.nn.L1Loss()
        self.mask_loss = torch.nn.BCELoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks, depths = batch
        e_masks, e_depths = self(images)

        mask_loss = self.mask_loss(e_masks, masks)
        depth_loss = self.depth_loss(e_depths * masks, depths)
        loss = mask_loss + depth_loss

        if batch_idx % self.log_loss_rate:
            self.log('train/loss', loss)
            self.log('train/mask_loss', mask_loss)
            self.log('train/depth_loss', depth_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, depths = batch
        e_masks, e_depths = self(images)

        e_depths = e_depths * masks
        
        mask_loss = self.mask_loss(e_masks, masks)
        depth_loss = self.depth_loss(e_depths, depths)

        e_masks = (e_masks > 0.5).float()
        inter = torch.count_nonzero(torch.logical_and(masks == 1, e_masks == 1), (1, 2))
        union = torch.count_nonzero(torch.add(masks, e_masks), (1, 2))
        mask_iou = inter / union

        return images, masks, depths, e_masks, e_depths, mask_loss, depth_loss, mask_iou

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def log_epoch_end(self, all_outputs, split):
        mask_loss_sum = 0
        depth_loss_sum = 0
        mask_iou_sum = 0
        for _, _, _, _, _, mask_loss, depth_loss, mask_iou in all_outputs:
            mask_loss_sum += mask_loss
            depth_loss_sum += depth_loss
            mask_iou_sum += mask_iou
        num_outputs = len(all_outputs)

        self.log(f'{split}/mean_mask_loss', mask_loss_sum / num_outputs)
        self.log(f'{split}/mean_depth_loss', depth_loss_sum / num_outputs)
        self.log(f'{split}/mean_mask_iou', mask_iou_sum / num_outputs)


        for i in range(min(len(all_outputs), self.num_images)):
            images, masks, depths, e_masks, e_depths, _, _, _ = all_outputs[i]

            image = Image.fromarray(np.uint8((images[0].permute(1, 2, 0) * 255).cpu().numpy()))
            mask = Image.fromarray(np.uint8((masks[0] * 255).cpu().numpy()))
            depth = Image.fromarray(np.uint8((depths[0] * 255).cpu().numpy()))
            e_mask = Image.fromarray(np.uint8((e_masks[0] * 255).cpu().numpy()))
            e_depth = Image.fromarray(np.uint8((e_depths[0] * 255).cpu().numpy()))
            
            self.logger.log_image(key=f'{split}_images/sample_{i}', images=[image, mask, depth, e_mask, e_depth])

    def validation_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'eval')

    def test_epoch_end(self, all_outputs):
        self.log_epoch_end(all_outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            DepthEstimatorDataset(self.data_dir, 'training'), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            DepthEstimatorDataset(self.data_dir, 'evaluation'), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            DepthEstimatorDataset(self.data_dir, 'training', self.test_steps), 
            batch_size=1, 
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )