import re
from jittor.dataset import Dataset
import numpy as np
import jittor as jt
from os import path as osp
from tqdm import tqdm

from led.utils.registry import DATASET_REGISTRY
from led.data.raw_utils import pack_raw_bayer


@DATASET_REGISTRY.register()
class PairedRAWDataset(Dataset):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.root_folder = opt['dataroot']
        self.postfix = opt.get('postfix', None)
        self.which_meta = opt.get('which_meta', 'gt')
        self.zero_clip = 0 if opt.get('zero_clip', True) else None
        self.use_patches = opt.get('use_patches', False)
        self.load_in_mem = opt.get('load_in_mem', False)
        if self.use_patches:
            assert self.load_in_mem == True
            self.patch_id_max = opt.get('patch_id_max', 8)
            self.patch_tplt = opt.get('patch_tplt', '_s{:03}')
        #     self.patch_tplt_re = opt.get('patch_tplt_re', '_s\d{3}')
        ## previous: only support npz with metadata
        # assert self.postfix == 'npz'

        self.lq_paths, self.gt_paths, self.ratios = [], [], []
        with open(opt['data_pair_list'], 'r') as data_pair_list:
            pairs = data_pair_list.readlines()
            for pair in pairs:
                lq, gt = pair.split(' ')[:2]
                gt = gt.rstrip('\n')
                
                # 清理路径：移除开头的 ./Sony/ 或 ./
                lq = lq.lstrip('./')
                gt = gt.lstrip('./')
                if lq.startswith('Sony/'):
                    lq = lq[5:]  # 移除 'Sony/'
                if gt.startswith('Sony/'):
                    gt = gt[5:]  # 移除 'Sony/'
                
                shutter_lq = float(re.search(r'_(\d+(\.\d+)?)s.', lq).group(0)[1:-2])
                shutter_gt = float(re.search(r'_(\d+(\.\d+)?)s.', gt).group(0)[1:-2])
                ratio = min(shutter_gt / shutter_lq, 300)
                if not self.use_patches:
                    self.lq_paths.append(osp.join(self.root_folder, lq))
                    self.gt_paths.append(osp.join(self.root_folder, gt))
                    self.ratios.append(ratio)
                else:
                    for i in range(1, 1 + self.patch_id_max):
                        self.lq_paths.append(osp.join(self.root_folder, self.insert_patch_id(lq, i, self.patch_tplt)))
                        self.gt_paths.append(osp.join(self.root_folder, self.insert_patch_id(gt, i, self.patch_tplt)))
                        self.ratios.append(ratio)
        if self.load_in_mem:
            get_data_path = lambda x: '.'.join([x, self.postfix]) if self.postfix is not None else x
            self.lqs = {
                get_data_path(data_path): \
                    self.depack_meta(get_data_path(data_path), self.postfix)
                for data_path in tqdm(set(self.lq_paths), desc='load lq metas in mem...')
            }
            self.gts = {
                get_data_path(data_path): \
                    self.depack_meta(get_data_path(data_path), self.postfix)
                for data_path in tqdm(set(self.gt_paths), desc='load gt metas in mem...')
            }

    @staticmethod
    def insert_patch_id(path, patch_id, tplt='_s{:03}'):
        exts = path.split('.')
        base = exts.pop(0)
        while exts[0] != 'ARW':
            base += '.' + exts.pop(0)
        base = base + tplt.format(patch_id)
        return base + '.' + '.'.join(exts)

    @staticmethod
    def depack_meta(meta_path, postfix='npz', to_tensor=True):
        if postfix == 'npz':
            meta = np.load(meta_path, allow_pickle=True)
            black_level = np.ascontiguousarray(meta['black_level'].copy().astype('float32'))
            white_level = np.ascontiguousarray(meta['white_level'].copy().astype('float32'))
            im = np.ascontiguousarray(meta['im'].copy().astype('float32'))
            wb = np.ascontiguousarray(meta['wb'].copy().astype('float32'))
            ccm = np.ascontiguousarray(meta['ccm'].copy().astype('float32'))
            meta.close()
        elif postfix == None:
            import rawpy
            import time
            ## using rawpy with retry mechanism for WSL I/O issues
            max_retries = 3
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    raw = rawpy.imread(meta_path)
                    raw_vis = raw.raw_image_visible.copy()
                    raw_pattern = raw.raw_pattern
                    wb = np.array(raw.camera_whitebalance, dtype='float32').copy()
                    wb /= wb[1]
                    ccm = np.array(raw.rgb_camera_matrix[:3, :3],
                                   dtype='float32').copy()
                    black_level = np.array(raw.black_level_per_channel,
                                           dtype='float32').reshape(4, 1, 1)
                    white_level = np.array(raw.camera_white_level_per_channel,
                                           dtype='float32').reshape(4, 1, 1)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Last attempt failed
                        raise RuntimeError(f"Failed to read {meta_path} after {max_retries} attempts: {e}")
            im = pack_raw_bayer(raw_vis, raw_pattern).astype('float32')
            raw.close()
        else:
            raise NotImplementedError

        if to_tensor:
            im = jt.array(im).float()
            black_level = jt.array(black_level).float()
            white_level = jt.array(white_level).float()
            wb = jt.array(wb).float()
            ccm = jt.array(ccm).float()

        return (im - black_level) / (white_level - black_level), \
                wb, ccm

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        gt_path = self.gt_paths[index]
        ratio = self.ratios[index]

        if self.postfix is not None:
            lq_path = '.'.join([lq_path, self.postfix])
            gt_path = '.'.join([gt_path, self.postfix])

        if not self.load_in_mem:
            lq_im, lq_wb, lq_ccm = self.depack_meta(lq_path, self.postfix)
            gt_im, gt_wb, gt_ccm = self.depack_meta(gt_path, self.postfix)
        else:
            lq_im, lq_wb, lq_ccm = self.lqs[lq_path]
            gt_im, gt_wb, gt_ccm = self.gts[gt_path]

        ### augment
        ## crop
        if self.opt['crop_size'] is not None:
            _, H, W = lq_im.shape
            crop_size = self.opt['crop_size']
            assert crop_size <= H and crop_size <= W
            if self.opt['phase'] == 'train':
                h_start = jt.randint(0, H - crop_size, (1,)).item()
                w_start = jt.randint(0, W - crop_size, (1,)).item()
            else:
                # center crop
                h_start = (H - crop_size) // 2
                w_start = (W - crop_size) // 2
            lq_im_patch = lq_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
            gt_im_patch = gt_im[:, h_start:h_start+crop_size, w_start:w_start+crop_size]
        else:
            lq_im_patch = lq_im
            gt_im_patch = gt_im
        ## flip + rotate
        if self.opt['phase'] == 'train':
            hflip = self.opt['use_hflip'] and jt.rand((1,)).item() < 0.5
            vflip = self.opt['use_rot'] and jt.rand((1,)).item() < 0.5
            rot90 = self.opt['use_rot'] and jt.rand((1,)).item() < 0.5
            if hflip:
                lq_im_patch = jt.flip(lq_im_patch, dims=[2])
                gt_im_patch = jt.flip(gt_im_patch, dims=[2])
            if vflip:
                lq_im_patch = jt.flip(lq_im_patch, dims=[1])
                gt_im_patch = jt.flip(gt_im_patch, dims=[1])
            if rot90:
                lq_im_patch = lq_im_patch.permute(0, 2, 1)
                gt_im_patch = gt_im_patch.permute(0, 2, 1)

            if self.opt.get('ratio_aug') is not None:
                ratio_range = self.opt['ratio_aug']
                rand_ratio = jt.rand((1,)).item() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
                ## TODO: maybe there are some over-exposed?
                gt_im_patch = gt_im_patch / ratio * rand_ratio
                ratio = rand_ratio

        lq_im_patch = jt.clamp(lq_im_patch * ratio, self.zero_clip, 1)
        gt_im_patch = jt.clamp(gt_im_patch, 0, 1)

        return {
            'lq': lq_im_patch,
            'gt': gt_im_patch,
            'ratio': jt.array(ratio).float(),
            'wb': gt_wb if self.which_meta == 'gt' else lq_wb,
            'ccm': gt_ccm if self.which_meta == 'gt' else lq_ccm,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)