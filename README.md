# Integrating Large Language Models with Robotic Grasp Planning: A Task-Oriented Approach

## Project Overview
This repository contains all the resources and notebooks for our project on enhancing robotic grasp planning through the integration of Large Language Models (LLMs) like OpenAI's GPT and Vision models. Our goal is to create a flexible, adaptive system that interprets both visual data and natural language prompts to generate effective grasp strategies tailored to specific tasks, improving human-robot interaction, grasp efficiency, and system adaptability.

## Installation
Clone this repository using:
```bash
git clone https://github.com/msheibani111/Integrating-Large-Language-Models-with-Robotic-Grasp-Planning-A-Task-Oriented-Approach.git
```


## Repository Structure
- `augmented_images_generation.ipynb`: Notebook for generating and augmenting images with markers.
- `Finetuning-Image-Captioning.ipynb`: Notebook for fine-tuning the Microsoft GIT model on the augmented image dataset.
- `grasp_point_generation.ipynb`: Notebook implementing the grasp point generation algorithm.
- `image_dataset_generation.ipynb`: Notebook detailing the creation of our custom image dataset using AI technologies.

## Usage
Each notebook contains detailed steps on how to execute the models and algorithms. Ensure you have the necessary hardware and software setup to handle the computational needs.

## Prompt and Instructions Examples
The `prompts_and_instructions.csv` file contains examples that are going to be used in the process of few-shot training a language model for the task of instruction generation.

## Image Dataset 
The Image dataset was generated using OpenAI's DALL-e model in the `image_dataset_generation.ipynb` notebook and was further refined using a simpler version of the Grasp Point Generation algorithm so that each image has the desired marking for fine-tuning the Captioning model.

## Grasp Point Generation
The grasp point generation process involves using the YOLOv8 model to detect objects in images and outline them with bounding boxes. The SAM (Segment Anything Model) further refines these outlines through precise segmentation. Within these segmented areas, potential grasp points are identified along a horizontal line, and their practicality is assessed using image captioning models. 

## Fine-Tuning Details
We utilized the Microsoft GIT model for image captioning, fine-tuned on our custom image dataset. Although initial results were not as expected due to data limitations, the process revealed that with a more robust dataset and additional tuning, the desired outcomes are achievable. Details and code for fine-tuning are provided in the `Finetuning-Image-Captioning.ipynb` notebook.



## Contributions
- Mohammad Sheibani (msheibani111@ut.ac.ir)
- Reihaneh Yourdkhani (r.yourdkhani@ut.ac.ir)
