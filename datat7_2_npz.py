import os
import yaml
import json

with open('config.yaml', 'r') as f:
	config = yaml.safe_load(f)

liste = [f for f in os.listdir('cub_icml') if not os.path.isdir(os.path.join('cub_icml',f))]
os.makedirs('Splits',exist_ok=True)
for file in liste:
  shutil.move(os.path.join('cub_icml',file), os.path.join('Splits',file))
liste = [f for f in os.listdir(config['text_path']) if not os.path.isdir(os.path.join(config['text_path'],f))]
for file in liste:
  os.remove(os.path.join(config['text_path'],file))


ebd_pth = config['embedding_path']
liste = [f for f in os.listdir(ebd_pth) if os.path.isdir(os.path.join(ebd_pth,f))]
import torchfile
import numpy as np
import json
files = []
Embeds = config['npz_path']
for folder in liste:
  full_folder = os.path.join(ebd_pth,folder)
  os.makedirs(os.path.join(Embeds,folder), exist_ok=True)
  for file in os.listdir(full_folder):
    if file.endswith('.t7'):
      file_path = os.path.join(full_folder, file)
      file_name = file.split('.')[0]
      example_data = torchfile.load(file_path)
      example_data = {
        k.decode() if isinstance(k, bytes) else k: v
        for k, v in example_data.items()
      }
      example_data['img'] = example_data['img'].decode('utf-8')
      arrays = {k: v for k, v in example_data.items() if isinstance(v, np.ndarray)}
      others = {k: v for k, v in example_data.items() if not isinstance(v, np.ndarray)}
      np.savez_compressed(os.path.join(Embeds,os.path.join(folder,file_name + '.npz')), **arrays)
      # print(os.path.join(Embeds,os.path.join(folder,file_name + '.json')))
      # print(others)
      with open(os.path.join(Embeds,os.path.join(folder,file_name + '.json')), "w") as f:
        json.dump(others, f)
      print(file_name)



