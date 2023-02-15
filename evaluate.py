from tensorflow import keras
from classification_dataset import DataGenerator
import argparse


def evaluation(args):
    img_size = args.test_image_size
    if isinstance(args.test_image_size, int):
        img_size = (args.test_image_size, args.test_image_size)

    model = keras.models.load_model(args.model_path)

    test_dataset = DataGenerator(
        images_path=args.test_path,
        label_csv=args.test_labels,
        dim=img_size)

    return model.evaluate(test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="trained model path / h5 file")
    parser.add_argument('--test_path', type=str, help="test dataset")
    parser.add_argument('--test_labels', help="csv file labels")
    parser.add_argument('--image_size', default=(224, 224), help="image size that model is trained on")
    args = parser.parse_args()
    evaluation(args)
