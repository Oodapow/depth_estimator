import torch
import pickle
import os
import cv2


def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    return ((1. - depth_map) * 255).astype('uint8')


def collate_fn(batch):
    images = torch.stack([torch.tensor(b[0]).permute(2, 0, 1).div(255.) for b in batch])
    masks = torch.stack([torch.tensor(b[1]).div(255.) for b in batch])
    depths = torch.stack([torch.tensor(b[2]).div(255.) for b in batch])

    return images, masks, depths


class DepthEstimatorDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, max_items=0):
        super().__init__()
        assert split in ['evaluation', 'training']
        self.path = path
        self.split = split
        anno_file = os.path.join(path, split, f'anno_{split}.pickle')
        with open(anno_file, 'rb') as f:
            self.sample_ids = list(pickle.load(f).keys())
        
        if max_items and max_items > len(self.sample_ids):
            self.sample_ids = self.sample_ids[:max_items]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        image = cv2.imread(os.path.join(self.path, self.split, 'color', '%.5d.png' % sample_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.path, self.split, 'mask', '%.5d.png' % sample_id))
        mask = cv2.cvtColor(mask.clip(max=1) * 255, cv2.COLOR_BGR2GRAY).astype('uint8')

        depth = cv2.imread(os.path.join(self.path, self.split, 'depth', '%.5d.png' % sample_id))
        depth = depth_two_uint8_to_float(depth[:, :, 2], depth[:, :, 1])

        return image, mask, depth

if __name__ == '__main__':

    data_path = '/home/oodapow/data/RHD_published_v2'

    dataset = DepthEstimatorDataset(data_path, 'evaluation')

    print(len(dataset))

    image, mask, depth = dataset[777]

    cv2.imwrite("depth.png", depth)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("image.png", image)