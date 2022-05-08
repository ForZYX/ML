from __future__ import absolute_import
import torch as t
from torch import nn
from torchvision.models import resnet34
from torchvision.ops import RoIPool

from object_detection.model.region_proposal_network import RegionProposalNetwork
from object_detection.model.faster_rcnn import FasterRCNN
from object_detection.utils import array_tool as at
from object_detection.utils.config import opt


def decom_resnet():
    model = resnet34(pretrained=False)
    if opt.caffe_pretrain:
        model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model.load_state_dict(t.load('F:/model_net/resnet34-333f7ec4.pth'))

    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features, classifier


class FasterRCNNRes34(FasterRCNN):
    """Faster R-CNN based on Res-34.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of layer3 in res34

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_resnet()

        rpn = RegionProposalNetwork(
            256, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = Res34RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNRes34, self).__init__(
            extractor,
            rpn,
            head,
        )


class Res34RoIHead(nn.Module):
    """Faster R-CNN Head for Res-34 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16
    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(Res34RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(512, n_class * 4)
        self.score = nn.Linear(512, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1).contiguous()
        # NOTE: important: yx->xy
        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x.cuda(), indices_and_rois)
        # pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7.view(-1, 512))
        roi_scores = self.score(fc7.view(-1, 512))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initializer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


if __name__ == "__main__":
    from object_detection.dataset.dataset import Dataset

    dataset = Dataset(opt)
    faster_rcnn = FasterRCNNRes34()
    img, bbox, label, scale = dataset.__getitem__(101)
    x = t.from_numpy(img).unsqueeze(0)
    img_size = x.shape[2:]
    feature_map = faster_rcnn.extractor(x)  # 1*256*38*42
    rpn_locs, rpn_scores, rois, roi_indices, anchor = faster_rcnn.rpn(feature_map, img_size, scale)
    roi_cls_locs, roi_scores = faster_rcnn.head(feature_map, rois, roi_indices)

