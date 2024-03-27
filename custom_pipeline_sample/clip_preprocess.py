''' Preprocessing function for CLIP's output to enable it to be fed into controlnet.

What CLIP does:
(from src/transformers/models/clip/modeling_clip.py - CLIPVisionEmbeddings)
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        # shape = [*, self.embed_dim, 16, 16]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        # shape = [*, 256, self.embed_dim]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # shape = [*, 1, self.embed_dim]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # class_embeds added as first of 256.
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

'''
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

import torch


class CLIPWrapper:
    def __init__(self, pretrained_model_name_or_path="openai/clip-vit-large-patch14") -> None:
        ''' Run the CLIP image embedding pipeline. '''
        self.model = CLIPVisionModel.from_pretrained(
            pretrained_model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path)

    def get_img_embed(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        return outputs.last_hidden_state

    @classmethod
    def reverse_img_embed(cls, clip_embed: torch.FloatTensor):
        ''' This function tries to convert torch.Size([*, 257, 1024]) to torch.Size([*, 1024, 16, 16]), 
        reversing the preprocessing that CLIP does.

        This is so that we could directly pass it (after zero convolutions) 
        into the third down block in the UNet since it expects a 16x16 size.

        We first remove the CLS dimension as that contains the overall semantics, 
        since we are more concerned about the spatiality.

        Then reverse the flatterning of the patches to form back the image.

        ### TODO: Unable to verify if the output resembles the spatial arrangement of the original image.

        '''
        # Check shape of img_embed = torch.Size([*, 257, 1024])
        assert clip_embed.shape[1] == 257 and clip_embed.shape[2] == 1024, (
            "Only image embedding from `openai/clip-vit-large-patch14` (224) supported."
        )

        # Get rid of class_embeds by removing first embedding
        patch_embed = clip_embed[:, 1:, :]
        # shape = [*, 256, 1024]

        transposed_embed = patch_embed.transpose(1, 2)
        # shape = [*, 1024, 256]

        # Reshape to 16 x 16
        reshaped_embed = torch.unflatten(transposed_embed, -1, (16, 16))
        # shape = [*, 1024, 16, 16]

        return reshaped_embed


if __name__ == "__main__":

    # Generate image embeddings

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    clip_pipeline = CLIPWrapper()
    
    last_hidden_state = clip_pipeline.get_img_embed(image)
    input_embed = CLIPWrapper.reverse_img_embed(last_hidden_state)

    print(last_hidden_state.shape)
    print(input_embed.shape)
