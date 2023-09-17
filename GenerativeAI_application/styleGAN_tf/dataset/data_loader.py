import os
import Image
import keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from functools import partial


class DogDataset(Sequence):
    def __init__(self, path, img_list, transform1=False):
        self.path=path
        self.transform1=transform1

        self.imgs=[]
        self.labels=[]

        classes=os.listdir(self.path)
        for cls in classes:
            cls_path= os.path.join(self.path, cls)
            imgs_idx = os.listdir(cls_path)
            for img_idx in imgs_idx:
                img_path= os.path.join(cls_path, img_idx)
                img  = Image.open(img_path)
                img_to_tensor = tf.convert_to_tensor(img)
                self.imgs.append(img_to_tensor)
                self.labels.append(cls)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img=self.imgs[idx]
        if self.transform1:
            img = self.imgTransform(img)
        label = self.labels[idx]
        return {'img':img, 'label':label}


import tensorflow as tf
import numpy as np
import keras

from functools import partial

def imgTransform(res, image):
    # resize & normalization 
    # only downsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def create_dataloader(data_path, res):

    # we use different batch size for different resolution, so larger image size
    # could fit into GPU memory. The keys is image resolution in log2
    batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
    # We adjust the train step accordingly
    train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}

    batch_size = batch_sizes[int(np.log2(res))]

    dl = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels='inferred', label_mode=None,
                                                        image_size=(64, 64), batch_size=32,  seed=123,
                                                        validation_split=0.2,subset="training",
                                                        shuffle=True)
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    # num_parallel_calls: 병렬연산 배치 수 , AUTOTUNE 사용시 사용가능 리소스를 기반으로 병렬 호출 수가 동적으로 설정
    # shuffle(): element를 임의로 shuffle. buffer_size만큼 element를 채운 후 임의의 요소들을 선택 후 다른 요소로 대치
    # - 완벽한 셔플링을 위해서 데이터셋의 전체 크기 이상의 버퍼 크기가 필요
    # print(dl.class_names)
    dl = dl.map(partial(imgTransform, res), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl