''' This script shows the mismatch between the embedding size.

This script runs stable_diffusion_image_vec.py under community pipelines.

Conclusion:
SD pipeline uses CLIPTextModel with padding/truncation up to 77 in the second dim.
Image embeddings come in the size of torch.Size([1, 257, 1024]) and afterCLIPVisionModelWithProjection,
It becomes torch.Size([1, 768]).
In CLIP, image and text embeddings are in the same embedding space after projection but in SD,
The embedding used is before projection, so there is no way of simply mapping the image embedding 
(before or after projection) to what SD uses.
'''
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection


from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel, AutoTokenizer


# # Generate image embeddings
# model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooled_output = outputs.pooler_output  # pooled CLS states

# print(last_hidden_state.shape) # torch.Size([1, 257, 1024])
# print(pooled_output.shape)  # torch.Size([1, 1024])



# # Generate image embeddings
# model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# image_embeds = outputs.image_embeds  # pooled CLS states

# print(last_hidden_state.shape) # torch.Size([1, 257, 1024])
# print(image_embeds.shape)  # torch.Size([1, 768])



# # Test out text embeddings to see shape
# model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# inputs = tokenizer(["The quick brown fox jumps over the lazy dog. They slept in the bushes in the forest."], padding=True, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# text_embeds = outputs.text_embeds

# print(last_hidden_state.shape)  # torch.Size([1, 12, 768]) | torch.Size([1, 21, 768]) (depends on length of input)
# print(text_embeds.shape)  # torch.Size([1, 768])



# # Test out text embeddings to see shape
# model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# inputs = tokenizer(["The quick brown fox jumps over the lazy dog. They slept in the bushes in the forest."], padding=True, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooled_output = outputs.pooler_output

# print(last_hidden_state.shape)  # torch.Size([1, 12, 768]) | torch.Size([1, 21, 768]) (depends on length of input)
# print(pooled_output.shape)  # torch.Size([1, 768])



from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", custom_pipeline="../examples/community/stable_diffusion_image_vec.py")
output = pipe(prompt="frog jumps over the moon")  # torch.Size([1, 77, 768])
# output = pipe(prompt_embeds=pooled_output)
# 77 is due to padding with zeros/truncating it.
# https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/text_encoder/config.json

print(output)
print(output.shape)
