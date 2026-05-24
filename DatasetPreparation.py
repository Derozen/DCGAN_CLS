import os
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
import torchfile
from PIL import Image
import yaml
import json

with open('Projet/config.yaml', 'r') as f:
	config = yaml.safe_load(f)

images_path = config['images_path']
embedding_path = config['npz_path']
text_path = config['text_path']
datasetDir = config['h5_file']

val_classes = open(config['val_split_path']).read().splitlines()
train_classes = open(config['train_split_path']).read().splitlines()
test_classes = open(config['test_split_path']).read().splitlines()

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

test_files = 'Test'
os.makedirs(test_files, exist_ok=True)

JH = 0
for _class in sorted(os.listdir(embedding_path)):
  split = ''
  if _class in train_classes:
    split = train
  elif _class in val_classes:
    split = valid
  elif _class in test_classes:
    split = test

  data_path = os.path.join(embedding_path, _class)
  txt_path = os.path.join(text_path, _class)
  for example, txt_file in zip(sorted(glob(data_path + "/*.npz")), sorted(glob(txt_path + "/*.txt"))):
    loaded = np.load(example)
    example_data = {key: loaded[key] for key in loaded}
    example_data = {**example_data, **json.load(open(example.split('.npz')[0] + '.json'))}

    img_path = example_data['img']
    embeddings = example_data['txt']
    example_name = img_path.split('/')[-1][:-4]
    f = open(txt_file, "r")
    txt = f.readlines()
    f.close()

    img_path = os.path.join(images_path, img_path)
    img = open(img_path, 'rb').read()

    txt_choice = np.random.choice(range(10), 5)

    embeddings = embeddings[txt_choice]
    txt = np.array(txt)
    txt = txt[txt_choice]
    dt = h5py.string_dtype(encoding='utf-8')

    for c, e in enumerate(embeddings):
      ex = split.create_group(example_name + '_' + str(c))
      ex.create_dataset('name', data=example_name)
      ex.create_dataset('img', data=np.void(img))
      ex.create_dataset('embeddings', data=e)
      ex.create_dataset('class', data=_class)
      ex.create_dataset('txt', data=str(txt[c]), dtype=dt)

    print(example_name)



