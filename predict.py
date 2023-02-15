import cv2
import argparse
import config as cfg
from classifier import Classifier


def main(args):
    image = cv2.imread(args.image_path)
    classifier = Classifier(args.model_path)
    pred = classifier.predict(image)
    if args.label_name:
        print([cfg.type_label[pred[0]], cfg.color_label[pred[1]]])
    else:
        print(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="onnx model weight")
    parser.add_argument('--image_path', type=str, help="image path")
    parser.add_argument('--label_name', type=bool, default=False, help="if true print label name ")
    args = parser.parse_args()
    main(args)
