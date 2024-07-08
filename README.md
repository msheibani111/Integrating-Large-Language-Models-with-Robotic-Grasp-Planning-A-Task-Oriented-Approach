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
To run any of the inference notebooks( `final_inference.ipynb`, `Evaluation.ipynb`) first add the `fine_tuned_blip_inference.py` and the `few_shot_instruction_generation.py` scripts to the work space and then run them.

## Prompt and Instructions Examples
The `prompts_and_instructions.csv` file contains examples that are going to be used in the process of few-shot training a language model for the task of instruction generation.

## Image Dataset 
The Image dataset was generated using OpenAI's DALL-e model in the `image_dataset_generation.ipynb` notebook and was further refined using a simpler version of the Grasp Point Generation algorithm so that each image has the desired marking for fine-tuning the Captioning model.

## Grasp Point Generation
The grasp point generation section describes a multi-step process to identify potential grasp points on objects for robotic manipulation. Initially, the YOLOv8 model detects and bounds objects in an input image, filtering out irrelevant ones. The Segment Anything Model (SAM) then segments these objects accurately. Within the segmented boundaries, random lines are drawn to create intersection points, which are visualized as potential grasp points. An image captioning model evaluates these points, filtering out those that are impractical or irrelevant based on the object's shape. This method combines advanced detection and segmentation techniques to enhance the adaptability and efficiency of robotic manipulation in various tasks.
The mentioned process of grasp point generation can be seen in the `grasp_point_generation.ipynb` alongside an example usage of the grasp point Grasp-Point Generation module.

## Fine-Tuning the Caption Generation model
We utilized the Microsoft GIT model for image captioning, fine-tuned on our custom image dataset. Although initial results were not as expected due to data limitations, the process revealed that with a more robust dataset and additional tuning, the desired outcomes are achievable. Details and code for fine-tuning are provided in the `Finetuning-Image-Captioning.ipynb` notebook.

Despite the initial promise, the GIT model did not perform as expected due to the limitations of our dataset. This setback led us to explore the BLIP model, specifically its large variant, renowned for its efficiency and effectiveness in image captioning tasks. To circumvent the challenge of our limited dataset and avoid extensive fine-tuning, we applied a Parameter-efficient Fine-tuning (PeFT) technique known as LoRA (Low-Rank Adaptation), which we fine-tuned roughly 10% of the model parameters using 16-bit half-precision. This approach significantly enhanced model performance without the need for extensive computational resources. Fine-tuning the `Salesforce/blip-image-captioning-large` model is done in the `Fine_tune_BLIP2_on_an_image_captioning_dataset.ipynb`, in the notebook alongside the fine-tuning process an inference of the fine-tuned model and its result can be seen.

## Instruction Generation model
The code in the `few_shot_instruction_generation.py` script demonstrate the process of generating Instructions based on the user input and the input image caption.
The script utilizes OpenAI API for GPT3.5-Turbo to generate instructions using a few-shot instruct-tuning method based on the inputs.

## Inference
To Generate Task-Oriented Grasp Points the `final_inference.ipynb` notebook utilizes all of the perviously mentioned modules to create the desired workflow. 
- The Caption Generation module uses the captioning model base version(before fine-tuning) to generate a caption for the input image, and passes the caption alongside the user input to the Instruction Generation module.
- The Instruction Generation module uses the perviously mentioned Instruction Generation model to create suitble instructions.
- The input image is passed to the Grasp Point Generation module, and the module generates several different grasp points.
- The generated grasp point images are then filterd using the Caption Generation module's output and the generated instructions using cosine similarity score.
- The grasp point with the highst similarity score is then reported as the best grasp point.

## Baselines
We have proposed two baseline methods for Grasp Point Generation to compare our method against: Random Grasp Point Selection and Geometric Grasp Point Selection. The Random Grasp Point Selection baseline involves the stochastic selection of grasp points uniformly across an object's surface, serving as a fundamental benchmark to compare against more sophisticated models. The Geometric Grasp Point Selection baseline bypasses task-oriented prompts and relies solely on the visual features of the object, using simple heuristics or geometric analysis to determine viable grasp points. This method isolates the impact of task-specific information provided by the language model, allowing for a rigorous assessment of the enhancements brought by integrating contextual language understanding into robotic systems.

## Evaluation
Finally we evaluated our method and the baseline methods on a test data set which consists of the following columns: input image, user input, and ground truth grasp point.
Our method itself outputs the caption of the Grasp Point image, but the baseline methods only output the Grasp Point image; to resolve this problem we used OpenAI's GPT4-o API, which can process visual data,  to generate captions for the outputs of the baseline methods, then the average cosine similaity score between all of these captions and the ground truth grasp points were calculated. Our evaluation process proved that our method is a significant improvement over the baseline methods when it comes to Task-Orinted Grasp Point Generation. 



## Contributions
- Mohammad Sheibani (msheibani111@ut.ac.ir)
- Reihaneh Yourdkhani (r.yourdkhani@ut.ac.ir)
