import clip
import torch
from PIL import Image
import tensorflow as tf

device = "cuda" if tf.test.is_gpu_available() else "cpu"

class SensorySystem:
    class ClipVisualSubsystem:
        def __init__(self):
            self.model, self.transform = clip.load('ViT-B/32', device=device)

        # TODO:
        # 1. Is there an tensorflow implementation of CLIP?
        # 2. Current implementation does one inference per frame. When going to
        #   multi-modal, will have to align the video with the audio & other senses
        #   This is To be done.
        def encode(self, visual_input):
            # Note: visual_input should be a batch of images, preferably in PIL format
            # If it's not, you'll need preprocessing to convert whatever format it's in to PIL

            # Convert the images to CLIP-compatible format
            images = [self.transform(img).unsqueeze(0) for img in visual_input]
            images = torch.cat(images).to(device)

            # Get the image features
            with torch.no_grad():
                image_features = self.model.encode_image(images)

            # Convert torch tensor to tf tensor if needed
            image_features = tf.convert_to_tensor(image_features.cpu().numpy())

            return image_features


    def __init__(self):
        # Initialize the sensory processing modules here
        self.visual_module = SensorySystem.ClipVisualSubsystem()
        self.auditory_module, self.somatosensory_module = None, None
        # self.visual_module is a CLIP encoder of the image seen


    def process(self, visual_input, auditory_input, somatosensory_input):
        # Process sensory inputs using individual modules
        visual_output = self.visual_module.encode(visual_input)
        auditory_output = self.auditory_module(auditory_input)
        somatosensory_output = self.somatosensory_module(somatosensory_input)

        # Combine the sensory outputs
        # Replace the combination logic with a separate fully connected 1 layer network
        # Generate the code for it using tf.layers.dense
        combined_output = tf.layers.dense(visual_output, auditory_output, somatosensory_output, activation=None)

        return combined_output