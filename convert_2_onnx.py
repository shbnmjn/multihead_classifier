from tensorflow import keras
import argparse
import tf2onnx


def convert_2_onnx(args):
    model = keras.models.load_model(args.model_path)
    model_proto, _ = tf2onnx.convert.from_keras(model, opset=13, output_path=args.onnx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, help="onnx file name")
    parser.add_argument('--model_path', type=str, help="trained weight path")

    args = parser.parse_args()
    convert_2_onnx(args)