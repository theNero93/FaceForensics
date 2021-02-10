import argparse
import gc

import cv2
import dlib
import torch
import sys
import os

from tqdm import tqdm

sys.path.append(os.getcwd())

from detect_from_video import get_boundingbox, predict_with_model
from network.models import model_selection
from src.util.data.data_loader import load_subset
from src.util.dotdict import Dotdict
from src.util.validate import calc_scores


def get_opt():
    opt = Dotdict()

    opt.model = 'all'
    opt.is_train = True
    opt.pretrained = True
    opt.checkpoints_dir = './out/checkpoints/faces'
    opt.continue_train = True
    opt.save_name = 'latest'
    opt.name = 'knn'
    # opt.dataset_path = './datasets/celeb-df-v2/images'
    opt.dataset_path = './datasets/forensic/images'
    opt.multiclass = False
    opt.resize_interpolation = 'bilinear'
    opt.load_size = -1
    opt.train_split = 'train'
    opt.train_size = 2500
    opt.val_split = 'val'
    opt.val_size = 100
    opt.test_split = 'test'

    return opt


def pred_image(model, face_detector, image, cuda):
    # Image size
    height, width = image.shape[:2]
    # 2. Detect with dlib
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces):
        # For now only take biggest face
        face = faces[0]

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]

        # Actual prediction using our model
        prediction, output = predict_with_model(cropped_face, model,
                                                cuda=cuda)
        return prediction


def test(model_path, cuda):
    gc.collect()
    torch.cuda.empty_cache()
    # model, *_ = model_selection(modelname='xception', num_out_classes=2)
    print(cuda)
    if model_path is not None:
        model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()
    face_detector = dlib.get_frontal_face_detector()
    opt = get_opt()
    test_img, test_label = load_subset(opt, opt.test_split, opt.load_size)
    y_pred, y_true = [], []
    for img, label in tqdm(zip(test_img, test_label), total=len(test_img)):
        pred = pred_image(model, face_detector, img, cuda)
        if type(pred) == int:
            y_pred.append(pred)
            y_true.append(label)

    acc, ap, auc = calc_scores(y_true, y_pred)[:3]
    print("Detection only worked on {} from {} Images".format(len(y_pred), len(test_label)))
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_path', '-mi', type=str,
                   default='./src/baselines/face_forensics/classification/models/face_detection/xception/all_raw.p')
    p.add_argument('--cuda', type=bool, default=True)
    args = p.parse_args()
    test(**vars(args))
