import argparse
import gc
from os.path import join

import cv2
import dlib
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
from detect_from_video import get_boundingbox, predict_with_model
from src.data.basic_dataset import BasicDataset
from src.model.transforms.transform_builder import create_basic_transforms
from src.util.validate import calc_scores, save_pred_to_csv


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


def test(args):
    gc.collect()
    torch.cuda.empty_cache()
    # model, *_ = model_selection(modelname='xception', num_out_classes=2)
    print(args.cuda)
    model = torch.load(args.model_path)
    print('Model found in {}'.format(args.model_path))

    if args.cuda:
        model = model.cuda()
    face_detector = dlib.get_frontal_face_detector()
    dataset_dir = join(args.root_dir, args.dataset)
    test_data = BasicDataset(root_dir=dataset_dir,
                             processed_dir=args.processed_dir,
                             crops_dir=args.crops_dir,
                             split_csv=args.split_csv,
                             seed=args.seed,
                             normalize=None,
                             transforms=create_basic_transforms(300),
                             mode='test')
    test_loader = DataLoader(test_data, batch_size=1)

    y_pred, y_true = [], []
    for img, label in tqdm(test_loader):
        pred = pred_image(model, face_detector, np.asarray(transforms.ToPILImage()(img.squeeze(0))), args.cuda)
        if type(pred) == float:
            y_pred.append(pred)
            y_true.append(label.item())

    acc, auc, loss = calc_scores(y_true, y_pred)[:3]
    print("Detection only worked on {} from {} Images".format(len(y_pred), len(test_loader)))
    print("Test: acc: {}; auc: {}; loss: {}".format(acc, auc, loss))
    if args.save_pred:
        save_pred_to_csv(y_true, y_pred, args.name, args.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for Training")
    args = parser.add_argument
    # Dataset Options
    args("--root_dir", default='/bigssd/datasets', help="root directory")
    args('--dataset', default='dfdc')
    args('--processed_dir', default='processed', help='directory where the processed files are stored')
    args('--crops_dir', default='crops', help='directory of the crops')
    args('--split_csv', default='folds.csv', help='Split CSV Filename')
    args('--seed', default=111, help='Random Seed')
    args('--name', default='bl_face_forensics')
    args('--save_pred', default=True)

    args('--model_path', '-mi', type=str,
         default='./src/baselines/face_forensics/classification/models/face_detection/xception/all_c23.p')
    args('--cuda', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())
