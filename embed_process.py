import os
import json
import clip
import torch
from PIL import Image
from tqdm import tqdm
import faiss
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device+" is in use")

model,preprocess = clip.load("ViT-B/32",device)

dataset_path = "/Users/lawrenceli/Desktop/CLIP/self-training/ImageNet-Mini"
images_path = os.path.join(dataset_path, "images") 
class_map_path = os.path.join(dataset_path,"imagenet_class_index.json")

with open(class_map_path,'r') as f:
    class_map = json.load(f)
    
text_input = torch.cat([clip.tokenize(f"a photo of a {c[1]}") for c in class_map.values()]).to(device)

all_image_features = []
all_text_features = text_input

def process_image(image_path):
    
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    
    image_features /= image_features.norm(dim=-1,keepdim=True)
    
    return image_features

def embeddings_saving():
    embeddings_save_path = os.path.join(dataset_path, "embeddings")
    
    if not os.path.exists(embeddings_save_path):
        os.makedirs(embeddings_save_path)
    
    torch.save(all_image_features, os.path.join(embeddings_save_path,'image_embeddings.pt'))
    
    torch.save(all_text_features, os.path.join(embeddings_save_path,'text_embeddings.pt'))
    
    print('Embeddings have been saved.')
    

for class_id, (folder_name,class_name) in tqdm(class_map.items(),desc="Processing Images"):
    folder_path = os.path.join(images_path,folder_name)
        
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            print('Unknown file type!')
            continue
        image_path = os.path.join(folder_path,image_name)
        image_features = process_image(image_path)

        all_image_features.append(image_features)

all_image_features = torch.cat(all_image_features,dim=0)

with torch.no_grad():
    all_text_features = model.encode_text(text_input)
    all_text_features /= all_text_features.norm(dim=-1,keepdim=True)
        
    similarity = (100.0 * all_image_features @ all_text_features.T).softmax(dim=-1)
        
embeddings_saving()

for i, image_features in enumerate(all_image_features):
    if i % 1000 == 0:
        values, indices = similarity[i].topk(5)
        print(f"\nImage {i} top predictions:")
        for value, index in zip(values, indices):
            class_id = str(index.item())
            print(f"{class_map[class_id][1]:>16s}: {100 * value.item():.2f}%")
            