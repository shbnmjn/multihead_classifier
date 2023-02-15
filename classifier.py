import onnxruntime
import cv2
import numpy as np
import config as cfg


class Classifier:
    def __init__(self, onnx_path, enable_gpu=False):
        self._model_file_name = onnx_path
        self._enable_gpu = enable_gpu
        self._session = onnxruntime.InferenceSession(self._model_file_name,
                                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        if not enable_gpu:
            self._session.set_providers(['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = tuple(self._session.get_inputs()[0].shape[2:4])
        self._output_name = [out_.name for out_ in self._session.get_outputs()]

    @staticmethod
    def __preprocess(image, input_size=(224, 224), color=(114, 114, 114)):

        shape = image.shape[:2]  # current shape [height, width]

        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        # Scale ratio (new / old)
        r = min(input_size[0] / shape[0], input_size[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = input_size[1] - new_unpad[0], input_size[0] - new_unpad[1]

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, r, (dw, dh)

    def predict(self, image):
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = Classifier.__preprocess(image)
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32)
        pred = self._session.run(self._output_name, {self._input_name: image})
        return [np.argmax(pred[0]), np.argmax(pred[1])]