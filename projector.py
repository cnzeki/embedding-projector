# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import argparse
import json

import numpy as np


def save_tensor_bytes(save_path, x):
    y = x.flatten()
    n = len(y)
    with open(save_path, 'wb') as f:
        for i in range(n):
            v = y[i]
            f.write(v)


def save_label_tsv(save_path, x):
    y = x.flatten()
    n = len(y)
    with open(save_path, 'w') as f:
        for i in range(n):
            v = y[i]
            f.write('%s\n' % (str(v)))


def create_sprite_image(images, image_size=(-1, -1)):
    import cv2
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    # image NHWC
    if isinstance(images, list):
        images = np.array(images)
    if image_size[0] > 0:
        img_h = image_size[0]
        img_w = image_size[1]
    else:
        img_h = images.shape[1]
        img_w = images.shape[2]
    img_c = images.shape[3]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, img_c)) * 255

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                this_img = cv2.resize(this_img, (img_w, img_h))
                this_img = this_img.reshape(img_w, img_h, img_c)
                if img_c > 1:
                    this_img = cv2.cvtColor(this_img, cv2.COLOR_RGB2BGR)
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w, :] = this_img

    return spriteimage, [img_h, img_w]


def write_image_embeddings(root, title, feats, labels, imgs=None, sprite_size=(-1, -1), mode='w+'):
    '''
    Write embedding data for the `Embedding Project` tool to visualize
    :param root: root dir of `Embedding Project` tool
    :param title: name of the tensor
    :param feats: embedding tensor NxDim
    :param labels: labels for each sample NxNumClasses
    :param [optional] imgs: images in format NHWC
    :param [optional] sprite_size: image sprite size
    :param mode: 'w' -- write, 'w+' -- update or append, '+' -- append
    :return: None
    '''
    import cv2
    prefix = 'oss_data/' + title
    try:
        os.makedirs(os.path.join(root, 'oss_data/'))
    except:
        pass
    tensorPath = prefix + '.bytes'
    metadataPath = prefix + '.tsv'
    if imgs is not None:
        imagePath = prefix + '.png'
        # save sprites
        sprite, singleImageDim = create_sprite_image(imgs, sprite_size)
        cv2.imwrite(os.path.join(root,imagePath), sprite)
    # save tensor
    save_tensor_bytes(os.path.join(root,tensorPath), feats)
    # save meta
    save_label_tsv(os.path.join(root,metadataPath), labels)
    # config
    config_path = os.path.join(root,'oss_data/oss_demo_projector_config.json')
    config = {"modelCheckpointPath": "Demo datasets", "embeddings": []}
    # update or append mode
    if mode == 'w+' or mode == '+':
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
    # new tensor item
    item = {
        "tensorName": title,
        "tensorShape": list(feats.shape),
        "tensorPath": tensorPath,
        "metadataPath": metadataPath,

    }
    if imgs is not None:
        item["sprite"] = {
            "imagePath": imagePath,
            "singleImageDim": list(singleImageDim)
        }
    embeddings = config["embeddings"]
    if mode == 'w+':
        # check existence
        id = -1
        for i in range(len(embeddings)):
            if embeddings[i]["tensorName"] == title:
                id = i
                break
        if id >= 0:
            embeddings[id] = item
        else:
            config["embeddings"].append(item)
    else:
        config["embeddings"].append(item)
    # write out
    with open(config_path, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)


def save_tensor_tsv(path, feats):
    with open(path, 'w') as f:
        num, dim = feats.shape
        for line in range(num):
            for d in range(dim):
                if d > 0:
                    f.write('\t')
                v = str(feats[line][d])
                f.write(v)
            f.write('\n')


def write_tsv_embeddings(prefix, feats, labels=None):
    '''
    Write a tensor (or meta) to a tsv file for the `Embedding Project` tool
    :param prefix: output file prefix
    :param feats: embedding tensor NxDim
    :param labels: meta data
    :return: None
    '''
    feat_path = prefix + '_data.tsv'
    save_tensor_tsv(feat_path, feats)
    if labels is None:
        return
    dims = len(labels.shape)
    label_path = prefix + '_meta.tsv'
    if dims == 1:
        save_label_tsv(label_path, labels)
    else:
        save_tensor_tsv(label_path, labels)


def _gen_fake_feats(num_class, total_size, feat_dim, var=0.2):
    num_each_class = total_size // num_class
    # generate centers
    centers = np.random.randn(num_class, feat_dim)
    feats = np.random.randn(total_size, feat_dim)
    labels = []
    for i in range(total_size):
        cid = i // num_each_class
        cid = min(cid, num_class-1)
        feats[i] = feats[i] * var + centers[cid]
        labels.append(cid)
    labels = np.array(labels)
    feats = np.float32(feats)
    return feats, labels


def demo_test_word(root):
    title = 'words'
    batch_size = 500
    feat_dim = 48
    color_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Violet', 'Brown', 'Black', 'Grey', 'White']
    num_colors = len(color_names)

    # Fake metas
    feats, labels = _gen_fake_feats(num_colors, batch_size, feat_dim)
    metas = [color_names[labels[i]] for i in range(batch_size)]
    metas = np.array(metas)
    # write tsv files
    #write_tsv_embeddings(title, feats, metas)
    write_image_embeddings(root, title, feats, metas)


def demo_test_image(root):
    import cv2
    title = 'images'
    num_class = 10
    total_size = 500
    feat_dim = 48
    feats, labels = _gen_fake_feats(num_class, total_size, feat_dim)
    # generate fake images
    imgs = []
    for i in range(total_size):
        cid = labels[i]
        img = np.ones((40, 40, 1), dtype=np.uint8) * 255

        ox = np.random.randint(0, 25)
        oy = np.random.randint(20, 35)
        cv2.putText(img, str(cid), (ox, oy), 0, 1, (0, 0, 0))
        imgs.append(img)
    imgs = np.stack(imgs)
    write_image_embeddings(root, title, feats, labels, imgs, sprite_size=(24, 24))



if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='projector', conflict_handler='resolve')
    parser.add_argument('--port', type=int, default=8000, help='server port')
    parser.add_argument('--root', default='.', type=str, help='projector root dir')
    parser.add_argument('--demo', default=False, type=bool, help='write demo data')
    args = parser.parse_args()
    if args.demo:
        demo_test_image(args.root)
        demo_test_word(args.root)
    # start server
    import sys
    import os
    if sys.version_info.major == 2:
        os.system('python -m SimpleHTTPServer %s' % args.port)
    else:
        os.system('python -m http.server %s' % args.port)
