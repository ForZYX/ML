import random
import matplotlib.pyplot as plt
import cv2
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import os
import numpy as np
import time
from matplotlib import patches

'''
检测是否是多个类别，以及图片的大小是否有不一样的情况
经查实没有！ 类别都为：mitosis      size：（1485,1663,3）
                train set -> 313
                test set -> 80
                train set(RGB)->    
                                    mean ->  [199.4395731  124.41596151 179.67500149]
                                    std ->  [36.27309073 55.1893073  36.42408735]
                                    
                train set(RGB)0_1->
                                    mean ->  [0.78211598 0.48790573 0.70460784]
                                    std ->  [0.14226725 0.21643485 0.1428573]
'''


def detect_name_size():
    path = '../data/train/xml'
    xmls = os.listdir(path)
    for xml in xmls:
        c_path = os.path.join(path, xml)
        root = parse(c_path)
        rootNode = root.documentElement
        objects = rootNode.getElementsByTagName('object')

        size = rootNode.getElementsByTagName('size')
        if size.getElementsByTagName('width')[0].childNodes[0].data != 1663:
            print(size.getElementsByTagName('width')[0].childNodes[0].data)

        if size.getElementsByTagName('height')[0].childNodes[0].data != 1485:
            print(size.getElementsByTagName('height')[0].childNodes[0].data)

        if size.getElementsByTagName('depth')[0].childNodes[0].data != 3:
            print(size.getElementsByTagName('depth')[0].childNodes[0].data)

        for b in objects:
            if (b.getElementsByTagName('name')[0].childNodes[0].data != 'mitosis'):
                print(b.getElementsByTagName('name')[0].childnode[0].data)


def read_img(img_path, dtype=np.uint8):
    """Read an image from a file.

        This function reads an image from given file. The image is HWC format and
        the range of its value is :math:`[0, 255]`.

        Args:
            path (str): A path of image file.

            dtype: The type of array. The default value is :obj:`~numpy.float32`.
        :return:
            ~numpy.ndarray: An image.
        """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(dtype)

    if img.ndim == 2:
        return img[np.newaxis]
    # return (H, W, C)
    return img.transpose((2,0,1))


def read_xml(path):
    """
        最终返回的是一个多重列表：一个大的列表里面包含有许多长度为4的list，
        每一个小的list都为[xmin, ymin, xmax, ymax]

        结果形如：[[frame1], [frame2],...]
    """

    root = parse(path)
    rootNode = root.documentElement
    frames = []
    objects = rootNode.getElementsByTagName('object')
    for b in objects:
        t = []
        t.append((int)(b.getElementsByTagName('xmin')[0].childNodes[0].data))
        t.append((int)(b.getElementsByTagName('ymin')[0].childNodes[0].data))
        t.append((int)(b.getElementsByTagName('xmax')[0].childNodes[0].data))
        t.append((int)(b.getElementsByTagName('ymax')[0].childNodes[0].data))
        frames.append(t)
    return np.array(frames)


def readDetailedXml(path, VOC_BBOX_LABEL_NAMES=('mitosis')):
    """
    最终返回bbox label difficult
    """

    root = ET.parse(path)
    bbox = list()
    label = list()
    difficult = list()

    for obj in root.findall('object'):
        difficult.append(int(obj.find('difficult').text))
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
            int(bndbox_anno.find(tag).text) - 1
            for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
        name = obj.find('name').text.lower().strip()
        label.append(VOC_BBOX_LABEL_NAMES.index(name))

    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    # When `use_difficult==False`, all elements in `difficult` are False.
    difficult = np.array(difficult, dtype=bool).astype(np.uint8)  # PyTorch don't support np.bool

    return bbox, label, difficult


def cal_trainSet_mean_std():
    """
    This method aims to find the dataset's mean and std

    The next method tries to find dataset's mean amd std(containing images from 0 to 1)

    answer channel = R G B
    :return:
        return dataset's mean and std
    """
    base_path = 'F:/data/mitosis_detection/train/img'
    channels_sum, channels_squared_sum, nums = np.zeros(3), np.zeros(3), len(os.listdir(base_path))
    for i in os.listdir(base_path):
        img_path = os.path.join(base_path, i)
        img = read_img(img_path)
        channels_sum += img.mean(axis=1).mean(axis=1)
        channels_squared_sum += (img ** 2).mean(axis=1).mean(axis=1)

    print('channels_sum -> ', channels_sum)
    mean = channels_sum / nums
    std = (channels_squared_sum / nums - mean ** 2) ** 0.5
    print('mean -> ', mean)
    print('std -> ', std)


def cal_trainSet_img2_0_1_mean_std():
    # 计算训练集数据每一个通达的均值和方差
    base_path = 'F:/data/mitosis_detection/train/img'
    channels_sum, channels_squared_sum, nums = np.zeros(3, dtype=np.float), np.zeros(3, np.float), len(
        os.listdir(base_path))
    for i in os.listdir(base_path):
        img_path = os.path.join(base_path, i)
        img = read_img(img_path) / 255.0
        channels_sum += img.mean(axis=1).mean(axis=1)
        channels_squared_sum += (img ** 2).mean(axis=1).mean(axis=1)

    print('channels_sum -> ', channels_sum)
    mean = channels_sum / nums
    std = (channels_squared_sum / nums - mean ** 2) ** 0.5
    print('mean -> ', mean)
    print('std -> ', std)


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

        The bounding boxes are expected to be packed into a two dimensional
        tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. The second axis represents attributes of
        the bounding box. They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        where the four attributes are coordinates of the top left and the
        bottom right vertices.

        Args:
            bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
                :math:`R` is the number of bounding boxes.

            in_size (tuple): A tuple of length 2. The height and the width
                of the image before resized.

            out_size (tuple): A tuple of length 2. The height and the width
                of the image after resized.

            !!!attention: in_size && out_size is (H, W)
        Returns:
            ~numpy.ndarray:
            Bounding boxes rescaled according to the given image shapes.


        Examples
        --------
        ::

            dataset_path = 'F:/data/mitosis_detection'
            img_path = os.path.join(dataset_path, 'train/img/H05_06Cb.png')
            xml_path = os.path.join(dataset_path, 'train/xml/H05_06Cb.xml')
            begin = time.time()
            img = read_img(img_path)
            bbox = read_xml(xml_path)
            draw_bbox(img, bbox)

            H, W, C = img.shape

            min_size = 600
            max_size = 1000

            scale1 = min_size / min(H, W)
            scale2 = max_size / max(H, W)
            scale = min(scale2, scale1)
            t = sktf.resize(img.transpose((2, 0, 1)), (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
            _, o_H, o_W = t.shape

            t_bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

            t_pose = t.transpose((1, 2, 0))

            draw_bbox(t_pose, t_bbox)

            end = time.time()
            print('time cost -> ', end - begin)
        """

    new_bbox = bbox.copy()

    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    new_bbox[:, 0] = new_bbox[:, 0] * x_scale
    new_bbox[:, 2] = new_bbox[:, 2] * x_scale
    new_bbox[:, 1] = new_bbox[:, 1] * y_scale
    new_bbox[:, 3] = new_bbox[:, 3] * y_scale

    return new_bbox


def draw_bbox(img, bbox):
    """
    use matplotlib to draw a picture, draw rects on img and show it!

    Args:

        img: img should be array format as (H W C)
        bbox: the labeled gt_box. They are (x_{min}, y_{min}, x_{max}, y_{max})

    Return:
        show the picture
    """

    # convert CHW to HWC
    img = img.transpose((1, 2, 0))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 画矩形框
    plt.imshow(img)
    currentAxis = plt.gca()  # 获取当前子图
    for box in bbox:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        currentAxis.add_patch(rect)
    plt.show()


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

        This method is mainly used together with image cropping.
        This method translates the coordinates of bounding boxes like
        :func:`data.util.translate_bbox`. In addition,
        this function truncates the bounding boxes to fit within the cropped area.
        If a bounding box does not overlap with the cropped area,
        this bounding box will be removed.

        The bounding boxes are expected to be packed into a two-dimensional
        tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. The second axis represents attributes of
        the bounding box. They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        where the four attributes are coordinates of the top left and the
        bottom right vertices.

        Args:
            bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
                :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            y_slice (slice): The slice of y-axis.
            x_slice (slice): The slice of x-axis.
            allow_outside_center (bool): If this argument is :obj:`False`,
                bounding boxes whose centers are outside of the cropped area
                are removed. The default value is :obj:`True`.
            return_param (bool): If :obj:`True`, this function returns
                indices of kept bounding boxes.

        Returns:
            numpy.ndarray or (numpy.ndarray, dict):

                If :obj:`return_param = False`, returns an array :obj:`bbox`.

                If :obj:`return_param = True`,
                returns a tuple whose elements are :obj:`bbox, param`.
                :obj:`param` is a dictionary of intermediate parameters whose
                contents are listed below with key, value-type and the description
                of the value.

                * **index** (*numpy.ndarray*): An array holding indices of used \
                    bounding boxes.
        Example:
        --------
        ::

            >>> crop_bb = np.array([101,251,300,450])
            >>> bbox = np.array([[50,150,250,360],[150,300,452,400]])

            >>> center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
            >>> mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)
            >>> print(mask)
                [True False]
        """
    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((l, t, r, b))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        # mask true when bbox's center in crop_bb, false when not.
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    # make new_bbox become a new correct bbox through cropping
    # first, we should use the max (xmin, ymin) and min (xmax, ymax)
    # then, each box should minus crop_bb(xmin, ymin)
    new_bbox = bbox.copy()
    new_bbox[:, :2] = np.maximum(new_bbox[:, :2], crop_bb[:2])
    new_bbox[:, 2:] = np.minimum(new_bbox[:, 2:], crop_bb[2:])
    new_bbox[:, :2] -= crop_bb[:2]
    new_bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (new_bbox[:, :2] < new_bbox[:, 2:]).all(axis=1))
    new_bbox = new_bbox[mask]

    if return_param:
        return new_bbox, {'index': np.flatnonzero(mask)}
    else:
        return new_bbox


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two-dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y-axis.
        x_offset (int or float): The offset along x-axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (x_offset, y_offset)
    out_bbox[:, 2:] += (x_offset, y_offset)

    return out_bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

        The bounding boxes are expected to be packed into a two-dimensional
        tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. The second axis represents attributes of
        the bounding box. They are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
        where the four attributes are coordinates of the top left and the
        bottom right vertices.

        Args:
            bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
                :math:`R` is the number of bounding boxes.
            size (tuple): A tuple of length 2. The height and the width
                of the image before resized.
            y_flip (bool): Flip bounding box according to a vertical flip of
                an image.
            x_flip (bool): Flip bounding box according to a horizontal flip of
                an image.

        Returns:
            numpy.ndarray:Bounding boxes flipped according to the given flips.
        """
    H, W = size
    new_bbox = bbox.copy()
    if y_flip:
        y_max = H - new_bbox[:, 1]
        y_min = H - new_bbox[:, 3]
        new_bbox[:, 1] = y_min
        new_bbox[:, 3] = y_max
    if x_flip:
        x_max = W - new_bbox[:, 0]
        x_min = W - new_bbox[:, 2]
        new_bbox[:, 0] = x_min
        new_bbox[:, 2] = x_max

    return new_bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


if __name__ == '__main__':
    dataset_path = 'F:/data/mitosis_detection'
    img_path = os.path.join(dataset_path, 'train/img/H12_01Ac.png')
    xml_path = os.path.join(dataset_path, 'train/xml/H12_01Ac.xml')
    '''
    test-example:   img: F:/data/mitosis_detection/train/img/H05_06Cb.png
                    xml: F:/data/mitosis_detection/train/xml/H05_06Cb.xml
    '''
    begin = time.time()

    img = read_img(img_path)
    bbox = read_xml(xml_path)
    draw_bbox(img, bbox)

    end = time.time()
    print('time cost -> ', end - begin)
