import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        confusion_matrix = self.confusion_matrix.copy()

        Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        confusion_matrix = self.confusion_matrix.copy()

        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        MIoU = np.nan_to_num(MIoU)
        print('MIoU:', MIoU)

        MIoU = MIoU[1:]
        MIoU = np.mean(MIoU)

        return MIoU

    def F1_score(self):
        confusion_matrix = self.confusion_matrix.copy()

        precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
        recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        F1 = (2 * precision * recall) / (precision + recall)
        print('recall:', recall)
        print('precision:', precision)

        F1 = np.nan_to_num(F1)
        print('F1:', F1)

        F1 = F1[1:]
        F1 = np.mean(F1)
        return F1

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
