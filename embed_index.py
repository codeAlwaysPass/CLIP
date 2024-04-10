import os
import torch
import json
import faiss
import numpy as np

def calculate_accuracy(query_labels,retrieved_indices,index_labels):
    correct_hits = 0
    
    for i,indices in enumerate(retrieved_indices):
        
        if query_labels[i] in [index_labels[idx] for idx in indices]:
            correct_hits += 1
    
    return correct_hits / len(query_labels)

image_embed = torch.load("/Users/lawrenceli/Desktop/CLIP/self-training/ImageNet-Mini/embeddings/image_embeddings.pt")
text_embed = torch.load("/Users/lawrenceli/Desktop/CLIP/self-training/ImageNet-Mini/embeddings/text_embeddings.pt")

image_embeddings_np = image_embed.numpy()
text_embeddings_np = text_embed.numpy()

dataset_path = "/Users/lawrenceli/Desktop/CLIP/self-training/ImageNet-Mini"
images_path = os.path.join(dataset_path, "images") 
class_map_path = os.path.join(dataset_path,"imagenet_class_index.json")

with open(class_map_path,'r') as f:
    class_map = json.load(f)

image_labels = []

for class_id, (folder_name,_) in class_map.items():
    folder_path = os.path.join(images_path,folder_name)
    images = [img for img in os.listdir(folder_path)]
    
    image_labels.extend([class_id]*len(images))

text_labels = [class_id for class_id, _ in class_map.items()]

dimension = image_embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings_np)

k = 5
D,I = index.search(text_embeddings_np,k)

accuracy = calculate_accuracy(text_labels,I,image_labels)

print(f"检索准确率: {accuracy:.4f}")

