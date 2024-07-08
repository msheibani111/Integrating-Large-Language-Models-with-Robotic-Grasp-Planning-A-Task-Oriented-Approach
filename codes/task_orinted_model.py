from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
scoring_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = scoring_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

"""## Find the best grasp score"""

def best_grasp_point(input_image_path,out_put_path,user_input):
  #get the instructions
  input_caption = generate_caption(Image.open(input_image_path),base_model)
  print(input_caption)
  instructions = generate_instruction(user_input,input_caption)
  print(instructions)
  #extract the grasping location from the instructions

  parts = instructions.split("grasping location:")

  task_details = parts[0].strip()

  grasping_location = "grasping location a " + parts[1].strip()

  #get the grasp points

  grasp_points = generate_images(output_path,image_path)

  #find the best grasp point
  grasp_scores = []
  for image in grasp_points:
    caption = generate_caption(image,fine_tuned_model)
    # Get embeddings for the caption and the grasping location
    caption_embedding = get_embedding(caption)
    grasping_location_embedding = get_embedding(grasping_location)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(caption_embedding, grasping_location_embedding)
    grasp_scores.append([image,caption, cosine_sim[0][0]])


  highest_score_item = max(grasp_scores, key=lambda x: x[1])

  #return the best grasping location
  return highest_score_item[1]

output_path="/content/drive/MyDrive/LLM final project/test output"
image_path = "/content/drive/MyDrive/LLM final project/test/gheichi.jpg"
user_input = "i want to cut a paper"

print(f"The best grsaping location for \'{user_input}'\ is:\n")
print(best_grasp_point(image_path,output_path,user_input))