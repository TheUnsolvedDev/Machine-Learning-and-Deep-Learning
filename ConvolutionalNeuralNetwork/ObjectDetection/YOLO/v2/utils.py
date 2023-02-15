import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import copy

import xml.etree.ElementTree as ET

from params import *


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(
                        path_to_image)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:

                        obj['name'] = attr.text

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def kmeans(boxes, k, dist=np.median, seed=1):
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for icluster in range(k):
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters, nearest_clusters, distances


def plot_cluster_result(plt, clusters, nearest_clusters, WithinClusterSumDist, wh):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters == icluster
        c = current_palette[icluster]
        plt.rc('font', size=8)
        plt.plot(wh[pick, 0], wh[pick, 1], "p",
                 color=c,
                 alpha=0.5, label="cluster = {}, N = {:6.0f}".format(icluster, np.sum(pick)))
        plt.text(clusters[icluster, 0],
                 clusters[icluster, 1],
                 "c{}".format(icluster),
                 fontsize=20, color="red")
        plt.title("Clusters")
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))


current_palette = list(sns.xkcd_rgb.values())


class ImageReader(object):
    def __init__(self, IMAGE_H, IMAGE_W, norm=None):
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.norm = norm

    def encode_core(self, image, reorder_rgb=True):
        # resize the image to standard size
        image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
        if reorder_rgb:
            image = image[:, :, ::-1]
        if self.norm is not None:
            image = self.norm(image)
        return (image)

    def fit(self, train_instance):
        if not isinstance(train_instance, dict):
            train_instance = {'filename': train_instance}

        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape
        if image is None:
            print('Cannot find ', image_name)

        image = self.encode_core(image, reorder_rgb=True)

        if "object" in train_instance.keys():

            all_objs = copy.deepcopy(train_instance['object'])

            # fix object's position and size
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return image
        return image, all_objs


def normalize(image):
    return image / 255.


class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1])
                        for i in range(int(len(ANCHORS)//2))]

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap(
            [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap(
            [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union

    def find(self, center_w, center_h):
        best_anchor = -1
        max_iou = -1
        shifted_box = BoundBox(0, 0, center_w, center_h)
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou
        return (best_anchor, max_iou)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        self.confidence = confidence
        self.set_class(classes)

    def set_class(self, classes):
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return (self.label)

    def get_score(self):
        return (self.classes[self.label])
    
def rescale_centerxy(obj,config):
    center_x = .5*(obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])
    center_y = .5*(obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
    return(center_x,center_y)

def rescale_centerwh(obj,config):
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W']) 
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H']) 
    return(center_w,center_h)


if __name__ == '__main__':
    train_image, seen_train_labels = parse_annotation(
        train_annot_folder, train_image_folder, labels=LABELS)
    print("N train = {}".format(len(train_image)))

    wh = []
    for anno in train_image:
        aw = float(anno['width'])
        ah = float(anno['height'])
        for obj in anno["object"]:
            w = (obj["xmax"] - obj["xmin"])/aw
            h = (obj["ymax"] - obj["ymin"])/ah
            temp = [w, h]
            wh.append(temp)

    wh = np.array(wh)
    print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))
    kmax = 11
    dist = np.mean
    results = {}
    for k in range(2, kmax):
        clusters, nearest_clusters, distances = kmeans(
            wh, k, seed=2, dist=dist)
        WithinClusterMeanDist = np.mean(
            distances[np.arange(distances.shape[0]), nearest_clusters])
        result = {"clusters":             clusters,
                  "nearest_clusters":     nearest_clusters,
                  "distances":            distances,
                  "WithinClusterMeanDist": WithinClusterMeanDist}
        print("{:2.0f} clusters: mean IoU = {:5.4f}".format(
            k, 1-result["WithinClusterMeanDist"]))
        results[k] = result

    # figsize = (15, 35)
    count = 1
    fig = plt.figure()
    for k in range(2, kmax):
        result = results[k]
        clusters = result["clusters"]
        nearest_clusters = result["nearest_clusters"]
        WithinClusterSumDist = result["WithinClusterMeanDist"]

        ax = fig.add_subplot(3, 3, count)
        plot_cluster_result(plt, clusters, nearest_clusters,
                            1 - WithinClusterSumDist, wh)
        count += 1
    plt.show()

    Nanchor_box = 4
    print(results[Nanchor_box]["clusters"])

    print("*"*30)
    print("Input")
    timage = train_image[0]
    for key, v in timage.items():
        print("  {}: {}".format(key, v))
    print("*"*30)
    print("Output")
    inputEncoder = ImageReader(
        IMAGE_H=IMG_SIZE[0], IMAGE_W=IMG_SIZE[1], norm=normalize)
    image, all_objs = inputEncoder.fit(timage)
    print("          {}".format(all_objs))
    plt.imshow(image)
    plt.title("image.shape={}".format(image.shape))
    plt.show()

    _ANCHORS01 = np.array([0.08285376, 0.13705531,
                           0.20850361, 0.39420716,
                           0.80552421, 0.77665105,
                           0.42194719, 0.62385487])
    print(".."*40)
    print("The three example anchor boxes:")
    count = 0
    for i in range(0, len(_ANCHORS01), 2):
        print("anchor box index={}, w={}, h={}".format(
            count, _ANCHORS01[i], _ANCHORS01[i+1]))
        count += 1
    print(".."*40)
    print("Allocate bounding box of various width and height into the three anchor boxes:")
    babf = BestAnchorBoxFinder(_ANCHORS01)
    for w in range(1, 9, 2):
        w /= 10.
        for h in range(1, 9, 2):
            h /= 10.
            best_anchor, max_iou = babf.find(w, h)
            print("bounding box (w = {}, h = {}) --> best anchor box index = {}, iou = {:03.2f}".format(
                w, h, best_anchor, max_iou))
