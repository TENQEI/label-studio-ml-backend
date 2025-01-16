from io import BytesIO
from os import getenv
from typing import List, Dict, Optional

import numpy as np
import requests
from PIL import Image
from label_studio_sdk.label_interface.objects import PredictionValue

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from gpr_bbox_predict_20250109 import MyInference, predict_image

model_path = getenv("MODEL_DIR", "model-20250109")

inference_instance = MyInference(model_path)


def get_image_from_url(url):
    # 发送GET请求获取图片
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from URL: {url}")

    # 将响应内容转换为字节流
    image_bytes = BytesIO(response.content)

    # 使用PIL打开图片
    image = Image.open(image_bytes)

    # 将图片转换为NumPy数组
    image_array = np.array(image)

    return image_array


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", model_path)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}
        # Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        predictions = []

        for task in tasks:
            data = get_image_from_url(task['data']['image'])
            result = predict_image(inference_instance, data[..., :3])
            predictions.append(PredictionValue(**{
                "model_version": "main_classification",
                "score": result['main_classification']['confidence'],
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "id": task['id'],
                        "source": task['data']['image'],
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [result['main_classification']['predicted_label']],
                            "rotation": 0,
                            "width": (result['bbox'][2] - result['bbox'][0]) * 100,
                            "height": (result['bbox'][3] - result['bbox'][1]) * 100,
                            "x": result['bbox'][0] * 100,
                            "y": result['bbox'][1] * 100
                        }
                    }
                ]
            }))

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        
        return ModelResponse(model_version=self.model_version, predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
