#!/usr/bin/env python

import argparse
import multiprocessing
import os.path as osp

import jsk_data


def download_data(*args, **kwargs):
    p = multiprocessing.Process(
            target=jsk_data.download_data,
            args=args,
            kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'rs_test'

    download_data(
        pkg_name=PKG,
        path='trained_data/mask_rcnn_resnet50_coco_trained.npz',
        url='https://drive.google.com/uc?id=19sciU40y_a3tN18QyLiQcWAuGc2hZw9p',
        md5='8e06483c0726acdb007ecbf503316a2d',
        quiet=quiet,
    )


if __name__ == '__main__':
    main()
