# import open3d as o3d
import torch
import numpy as np
import pandas as pd
import random
import os

from util.config import cfg
cfg.task = 'test'
import util.utils as utils
import util.eval as eval
from lib.pointgroup_ops.functions import pointgroup_ops


class InsSegTest():
    def __init__(self):
        self.semantic_label_idx = [0, 1]

        random.seed(cfg.test_seed)
        np.random.seed(cfg.test_seed)
        torch.manual_seed(cfg.test_seed)
        torch.cuda.manual_seed_all(cfg.test_seed)

        torch.backends.cudnn.enabled = False

        exp_name = cfg.config.split('/')[-1][:-5]
        model_name = exp_name.split('_')[0]

        if model_name == 'InsSegNet':
            from BuildingInsSeg.model import InstanceSegPipline as Network
            from BuildingInsSeg.model import model_fn_decorator
        else:
            print("Error: no model version " + model_name)
            exit(0)
        self.model = Network(cfg)

        use_cuda = torch.cuda.is_available()
        assert use_cuda
        self.model = self.model.cuda()

        self.model_fn = model_fn_decorator(test=True)

        # load model
        utils.checkpoint_restore(self.model, './exp', cfg.config.split('/')[-1][:-5],
                                 use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)

    def dataloader(self, data_path):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []
        batch_offsets = [0]
        scene_names = os.path.basename(data_path).strip('.txt')

        data = pd.read_csv(data_path, delimiter=' ', header=None).to_numpy().astype(np.float32)
        xyz_origin, rgb, label, instance_label = data[:, :3], data[:, 3:6], data[:, 6], data[:, 7]

        # scale
        xyz = xyz_origin * cfg.scale

        # offset
        xyz -= xyz.min(0)

        # merge the scene to the batch
        batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

        locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()], 1))
        locs_float.append(torch.from_numpy(xyz_origin))
        feats.append(torch.from_numpy(rgb))
        labels.append(torch.from_numpy(label))
        instance_labels.append(torch.from_numpy(instance_label))

        labels = torch.cat(labels, 0).long()  # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()  # long (N)

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, cfg.batch_size, cfg.mode)

        return [{'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                'labels': labels, 'instance_labels': instance_labels, 'scene_names': scene_names}]


    def test(self, data_path, epoch=384):
        '''
        :param data_path: 测试文件路径（.txt文件）
        :param epoch: 默认384不用管
        :return: 返回一个字典，里面有所有你想要的东西：
            xyz_np          输入点云的三维坐标 numpy数组           (N, 3) float
            rgb_np          输入点云的rgb颜色值 numpy数组          (N, 3) float (0~255)
            sem_gt_np       语义gt numpy数组                     (N,) int (0~1)
            ins_gt_np       实例gt numpy数组                     (N,) int (-100 for ignored label, 0~nInstance)
            sem_pred_np     语义预测 numpy数组                    (N,) int (0~1)
            ins_pred_np     实例预测 numpy数组                    (N,) int (-100 for ignored label, 0~nProposal)
            nInstance       实例gt的个数                          int
            nProposal       实例预测结果proposal的个数              int
            APs             实例预测评估结果(AP, AP50, AP25)        tuple(float, float, float)
            sem_oa          语义分割总体准确率                      float
            sem_miou        语义分割平均IoU                        float
        '''
        if not os.path.exists(data_path):
            print("file not exists!")
            exit(0)

        dataloader = self.dataloader(data_path)

        with torch.no_grad():
            model = self.model.eval()

            matches = {}

            xyz_np = None
            rgb_np = None
            sem_gt_np = None
            ins_gt_np = None
            sem_pred_np = None
            ins_pred_np = None
            nInstance = None
            nProposal = None
            APs = None
            sem_oa = None
            sem_miou = None

            for i, batch in enumerate(dataloader):
                # unpack data
                xyz_np = dataloader[0]['locs_float'].numpy()
                rgb_np = dataloader[0]['feats'].numpy()
                rgb_np = (rgb_np + 1) * 127.5
                sem_gt_np = dataloader[0]['labels'].numpy()
                ins_gt_np = dataloader[0]['instance_labels'].numpy()
                nInstance = np.unique(ins_gt_np, axis=-1).shape[0] - 1  # ignore -100

                # inference
                preds = self.model_fn(batch, model, epoch)

                # decode results for evaluation
                N = batch['feats'].shape[0]
                test_scene_name = batch['scene_names']
                print(test_scene_name)
                semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
                semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
                if (epoch > cfg.prepare_epochs):
                    scores = preds['score']   # (nProposal, 1) float, cuda
                    scores_pred = torch.sigmoid(scores.view(-1))

                    proposals_idx, proposals_offset = preds['proposals']
                    # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
                    # proposals_offset: (nProposal + 1), int, cpu
                    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device)
                    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                    # (nProposal, N), int, cuda

                    semantic_id = torch.tensor(self.semantic_label_idx, device=scores_pred.device) \
                        [semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long
                    # semantic_id_idx = semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]

                    # score threshold
                    score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                    scores_pred = scores_pred[score_mask]
                    proposals_pred = proposals_pred[score_mask]
                    semantic_id = semantic_id[score_mask]
                    # semantic_id_idx = semantic_id_idx[score_mask]

                    # npoint threshold
                    proposals_pointnum = proposals_pred.sum(1)
                    npoint_mask = (proposals_pointnum >= cfg.TEST_NPOINT_THRESH)
                    scores_pred = scores_pred[npoint_mask]
                    proposals_pred = proposals_pred[npoint_mask]
                    semantic_id = semantic_id[npoint_mask]

                    clusters = proposals_pred
                    cluster_scores = scores_pred
                    cluster_semantic_id = semantic_id
                    nclusters = clusters.shape[0]
                    nProposal = nclusters

                    # prepare for evaluation
                    if cfg.eval:
                        pred_info = {}
                        pred_info['conf'] = cluster_scores.cpu().numpy()
                        pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                        pred_info['mask'] = clusters.cpu().numpy()
                        gt_file = os.path.join('exp', cfg.split + '_gt', test_scene_name + '.txt')
                        gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)

                        matches[test_scene_name] = {}
                        matches[test_scene_name]['gt'] = gt2pred
                        matches[test_scene_name]['pred'] = pred2gt

                        if cfg.split == 'val':
                            matches[test_scene_name]['seg_gt'] = batch['labels']
                            matches[test_scene_name]['seg_pred'] = semantic_pred

                # save files
                if cfg.save_semantic:
                    sem_pred_np = semantic_pred.cpu().numpy()

                ins_pred_np = np.array([-100] * N)
                if(epoch > cfg.prepare_epochs and cfg.save_instance):
                    for proposal_id in range(nclusters):
                        clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                        ins_pred_np[clusters_i == 1] = proposal_id

            # evaluation
            if cfg.eval:
                ap_scores = eval.evaluate_matches(matches)
                avgs = eval.compute_averages(ap_scores)
                # eval.print_results(avgs)
                APs = (avgs["all_ap"], avgs["all_ap_50%"], avgs["all_ap_25%"])

            # evaluate semantic segmantation accuracy and mIoU
            if cfg.split == 'val':
                seg_accuracy = evaluate_semantic_segmantation_accuracy(matches)
                sem_oa = seg_accuracy.cpu().item()
                iou_list = evaluate_semantic_segmantation_miou(matches)
                iou_list = torch.tensor(iou_list)
                miou = iou_list.mean()
                sem_miou = miou.cpu().item()

            return {'xyz_np': xyz_np, 'rgb_np': rgb_np, 'sem_gt_np':sem_gt_np, 'ins_gt_np':ins_gt_np,
                    'sem_pred_np': sem_pred_np, 'ins_pred_np': ins_pred_np, 'nInstance': nInstance,
                    'nProposal': nProposal, 'APs': APs, 'sem_oa': sem_oa, 'sem_miou': sem_miou}

def evaluate_semantic_segmantation_accuracy(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    correct = (seg_gt_all[seg_gt_all != -100] == seg_pred_all[seg_gt_all != -100]).sum()
    whole = (seg_gt_all != -100).sum()
    seg_accuracy = correct.float() / whole.float()
    return seg_accuracy

def evaluate_semantic_segmantation_miou(matches):
    seg_gt_list = []
    seg_pred_list = []
    for k, v in matches.items():
        seg_gt_list.append(v['seg_gt'])
        seg_pred_list.append(v['seg_pred'])
    seg_gt_all = torch.cat(seg_gt_list, dim=0).cuda()
    seg_pred_all = torch.cat(seg_pred_list, dim=0).cuda()
    assert seg_gt_all.shape == seg_pred_all.shape
    iou_list = []
    for _index in seg_gt_all.unique():
        if _index != -100:
            intersection = ((seg_gt_all == _index) &  (seg_pred_all == _index)).sum()
            union = ((seg_gt_all == _index) | (seg_pred_all == _index)).sum()
            iou = intersection.float() / union
            iou_list.append(iou)

    return iou_list


if __name__ == '__main__':
    t = InsSegTest()
    result = t.test(data_path='exp/6_wanxia_0_66.txt')
    print(result)

    # for test
    # xyz = result['xyz_np']
    # rgb = result['rgb_np'].astype(np.int)
    #
    # sem_gt_np = result['sem_gt_np']
    # ins_gt_np = result['ins_gt_np']
    # ins_gt_np[ins_gt_np != -100] += 1
    # ins_gt_np[ins_gt_np == -100] = 0
    #
    # sem_pred_np = result['sem_pred_np']
    # ins_pred_np = result['ins_pred_np']
    # ins_pred_np[ins_pred_np != -100] += 1
    # ins_pred_np[ins_pred_np == -100] = 0
    #
    # colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(50)])
    # sem_gt_color = colors[sem_gt_np]
    # ins_gt_color = colors[ins_gt_np]
    # ins_gt_color[0][0] = ins_gt_color[0][1] = ins_gt_color[0][2] = 0
    # sem_pred_color = colors[sem_pred_np]
    # ins_pred_color = colors[ins_pred_np]
    # ins_pred_color[0][0] = ins_pred_color[0][1] = ins_pred_color[0][2] = 0
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
    # o3d.visualization.draw_geometries([pcd])
    #
    # pcd.colors = o3d.utility.Vector3dVector(sem_gt_color / 255)
    # o3d.visualization.draw_geometries([pcd])
    #
    # pcd.colors = o3d.utility.Vector3dVector(sem_pred_color / 255)
    # o3d.visualization.draw_geometries([pcd])
    #
    # pcd.colors = o3d.utility.Vector3dVector(ins_gt_color / 255)
    # o3d.visualization.draw_geometries([pcd])
    #
    # pcd.colors = o3d.utility.Vector3dVector(ins_pred_color / 255)
    # o3d.visualization.draw_geometries([pcd])


