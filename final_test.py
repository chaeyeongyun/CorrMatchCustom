import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from util.dist_helper import setup_distributed
from model.semseg.deeplabv3plus import DeepLabV3Plus

from torch.utils.data import DataLoader
import yaml
from dataset.semi import SemiDataset
from util.utils import AverageMeter, intersectionAndUnion

from image_utils import save_img, make_test_detailed_img

class Measurement:
    def __init__(self, num_classes:int, ignore_idx=None) :
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def _make_confusion_matrix(self, pred:np.ndarray, target:np.ndarray):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): segmentation model's prediction score matrix
            target (numpy.ndarray): label
            num_classes (int): the number of classes
        """
        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        # prediction score to label
        pred_label = pred.argmax(axis=1) # (N, H, W)
        
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        # num_classes * gt + pred = category
        cats = self.num_classes * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, self.num_classes, self.num_classes)) # (N, 3, 3)
        return conf_mat
    
    def accuracy(self, pred, target):
        '''
        Args:
            pred: (N, C, H, W), ndarray
            target : (N, H, W), ndarray
        Returns:
            the accuracy per pixel : acc(int)
        '''
        N = pred.shape[0]
        pred = pred.argmax(axis=1) # (N, H, W)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
        target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
        
        if self.ignore_idx != None:
             not_ignore_idxs = np.where(target != self.ignore_idx) # where target is not equal to ignore_idx
             pred = pred[not_ignore_idxs]
             target = target[not_ignore_idxs]
             
        return np.mean(np.sum(pred==target, axis=-1)/pred.shape[-1])
    
    def miou(self, conf_mat:np.ndarray):
        iou_list = []
        sum_col = np.sum(conf_mat, -2) # (N, 3)
        sum_row = np.sum(conf_mat, -1) # (N, 3)
        for i in range(self.num_classes):
            batch_mean_iou = np.mean(conf_mat[:, i, i] / (sum_col[:, i]+sum_row[:, i]-conf_mat[:, i, i]+1e-8))
            iou_list.append(batch_mean_iou)
        iou_ndarray = np.array(iou_list)
        miou = np.mean(iou_ndarray)
        return miou, iou_list
    
    def precision(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_col = np.sum(conf_mat, -2)# (N, num_classes) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        precision_per_class = np.mean(np.array([conf_mat[:, i, i]/ (sum_col[:, i]+1e-7) for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        # multi class에 대해 recall / precision을 구할 때에는 클래스 모두 합쳐 평균을 낸다.
        mprecision = np.mean(precision_per_class)
        return mprecision, precision_per_class

    def recall(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_row = np.sum(conf_mat, -1)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        recall_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_row[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        mrecall = np.mean(recall_per_class)
        return mrecall, recall_per_class
    
    def f1score(self, recall, precision):
        return 2*recall*precision/(recall + precision)
    
    def measure(self, pred:np.ndarray, target:np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(pred, target)
        miou, iou_list = self.miou(conf_mat)
        precision, _ = self.precision(conf_mat)
        recall, _ = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, miou, iou_list, precision, recall, f1score
        
    __call__ = measure
    
def listmean(l:list):
    ret = 0
    for i in range(len(l)):
        ret += l[i]
    ret /= len(l)
    return ret
def evaluate(model, loader, mode, cfg, checkpoint_path):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    img_ret = []
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*cfg["nclass"], dtype=np.float64)
    measurement = Measurement(cfg["nclass"])
    with torch.no_grad():
        for img, mask, ids, img_ori in loader:
            img = img.cuda()
            b, _, h, w = img.shape
            res = model(img)
            pred = res['out'].argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            # dist.all_reduce(reduced_intersection)
            # dist.all_reduce(reduced_union)
            # dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(res["out"].detach().cpu().numpy(), mask.detach().cpu().numpy())
            test_acc += acc_pixel
            test_miou += batch_miou
            iou_per_class += iou_ndarray
            
            test_precision += precision
            test_recall += recall
            test_f1score += f1score
            viz = make_test_detailed_img(img_ori.detach().cpu().numpy()/255, res['out'].detach().cpu().numpy(), \
                    mask.detach().cpu().numpy())
            img_ret.append([viz, ids[0]])
            


    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    return_dict['iou_class'] = iou_class
    return_dict['mIOU'] = mIOU

    test_acc = test_acc / len(loader)
    # test_miou = test_miou / len(loader)
    test_ious = np.round((iou_per_class / len(loader)), 5).tolist()
    test_miou = listmean(test_ious)
    test_precision /= len(loader)
    test_recall /= len(loader)
    # test_f1score /= len(loader)
    test_f1score = (2 * test_precision * test_recall) / (test_precision + test_recall)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.4f, Test miou: %.4f" % (checkpoint_path,  test_acc, test_miou)       
    result_txt += f"\niou per class {list(map(lambda x: round(x, 4), test_ious))}"
    result_txt += f"\nprecision : {test_precision:.4f}, recall : {test_recall:.4f}, f1score : {test_f1score:.4f} " 
    print(result_txt)
    result_save_path = os.path.join(".", "test_save_files", cfg["dataset"]+"_test")
    os.makedirs(result_save_path, exist_ok=True)
    with open(os.path.join(result_save_path,"result.txt"), "w") as f:
        f.write(result_txt)

    return return_dict, img_ret
def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default="./configs/CWFID_percent30.yaml")
    parser.add_argument('--checkpoint_path', type=str, default="D:/CorrMatch/save_file/CWFID_percent301/resnet50_35.471.pth")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()
    # setup_distributed(port=args.port)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    model = DeepLabV3Plus(cfg)
    model.load_state_dict(torch.load(args.checkpoint_path))
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    local_rank = 0
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=False)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test')
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    model.eval()
    res_val, img_ret = evaluate(model, valloader, 'original', cfg, args.checkpoint_path)
    mIOU = res_val['mIOU']
    iou_class = res_val['iou_class']
    print(mIOU)
    print(iou_class)


if __name__ == '__main__':
    main()
