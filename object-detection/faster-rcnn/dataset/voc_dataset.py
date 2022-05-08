import os
import xml.etree.ElementTree as ET

import numpy as np

from object_detection.dataset.util import read_img, readDetailedXml


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

        .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

        The index corresponds to each image.

        When queried by an index, if :obj:`return_difficult == False`,
        this dataset returns a corresponding
        :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
        This is the default behaviour.
        If :obj:`return_difficult == True`, this dataset returns corresponding
        :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
        that indicates whether bounding boxes are labeled as difficult or not.

        The bounding boxes are packed into a two-dimensional tensor of shape
        :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
        the image. The second axis represents attributes of the bounding box.
        They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`, where the
        four attributes are coordinates of the top left and the bottom right
        vertices.

        The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
        :math:`R` is the number of bounding boxes in the image.
        The class name of the label :math:`l` is :math:`l` th element of
        :obj:`VOC_BBOX_LABEL_NAMES`.

        The array :obj:`difficult` is a one dimensional boolean array of shape
        :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
        If :obj:`use_difficult` is :obj:`False`, this array is
        a boolean array with all :obj:`False`.

        The type of the image, the bounding boxes and the labels are as follows.

        * :obj:`img.dtype == numpy.float32`
        * :obj:`bbox.dtype == numpy.float32`
        * :obj:`label.dtype == numpy.int32`
        * :obj:`difficult.dtype == numpy.bool`

        Args:
            data_dir (string): Path to the root of the training data.
                i.e. "'F:/data/mitosis_detection'"
            split ({'train', 'val', 'test'}): Select a split of the
                dataset. :obj:`test` split is only available for
                2007 dataset.
            use_difficult (bool): If :obj:`True`, use images that are labeled as
                difficult in the original annotation.
            return_difficult (bool): If :obj:`True`, this dataset returns
                a boolean array
                that indicates whether bounding boxes are labeled as difficult
                or not. The default value is :obj:`False`.

        """
    # base_path = 'F:/data/mitosis_detection'
    def __init__(self, data_dir, split='train',
                 use_difficult=False, return_difficult=False):
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = ('mitosis')

        self.dataset = os.path.join(data_dir, split)
        self.imgs = os.listdir(os.path.join(self.dataset, "img"))
        self.xmls = os.listdir(os.path.join(self.dataset, "xml"))

    def __len__(self):
        return len(self.imgs)

    def getVOC_BBOX_LABEL_NAMES(self):
        return self.label_names

    def __getitem__(self, item):
        """Returns the i-th example.

            Returns a color image and bounding boxes. The image is in CHW format.
            The returned image is RGB.

            Args:
                i (int): The index of the example.

            Returns:
                tuple of an image and bounding boxes
        """
        id = self.imgs[item].split(".")[0]

        img_path = os.path.join(self.dataset, "img", self.imgs[item])
        xml_path = os.path.join(self.dataset, "xml", id+".xml")
        img = read_img(img_path, dtype=np.float32)
        bbox, label, difficult = readDetailedXml(xml_path, self.label_names)

        return img, bbox, label, difficult
