# Running inference on a Model
The goal of this repository should be to verify the functionality of the already trained T5 model on the given dataset. We will use the raw dataset that is available in TensorFlow Datasets as "wmt_t2t_translate".

### Časová osa:
- [x] přiřazení osob, linky na github, data vystoupení (17.10.2023)
- [ ] poslední vystoupení (11.12.2023)
- [ ] náhradní termíny (3.1.,8.1.)
- [ ] udělování KZ (11.1)

# T5X
Link: https://github.com/google-research/t5x </br>
</br>
T5X is a modular, composable, research-friendly framework for high-performance, configurable, self-service training, evaluation, and inference of sequence models (starting with language) at many scales.

It is essentially a new and improved implementation of the T5 codebase (based on Mesh TensorFlow) in JAX and Flax. To learn more, see the T5X Paper.

Below is a quick start guide for training models with TPUs on Google Cloud. For additional tutorials and background, see the complete documentation.

# T5 - The solution I used, T5X - broken modules.
Text-to-text-transfer-transformer (T5): https://github.com/google-research/text-to-text-transfer-transformer.git</br>
Instructions from hugging face: https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/tensorflow/translation.ipynb

#### This repository contains a failed attempt to run inference on an already trained T5X model with the "wmt_t2t_translate" TensorFlow dataset, based on which I used the older T5 with HuggingFace's procedure.

### T5X Startup Errors:
- incomplete documentation
- missing module - ``` t5x.infer_unfragmented ``` which is used to start inference, it was deleted from the repository

</br>

## Postup: 
Naklonování repozitáře ```git clone https://github.com/google-research/t5x.git``` ✅

### Step 1: Installation of necessary libraries
Installation of libraries that are often used in the development and evaluation of natural language processing models.
``` 
pip install transformers datasets evaluate sacrebleu
```
### Step 2: Load OPUS Books dataset </br>
Test the English-French T5 subset of the OPUS Books dataset and translate English text into French.

Load the English-French subset of the OPUS Books dataset from the Datasets library:
```
from datasets import load_dataset
books = load_dataset("opus_books", "en-fr")
```
Split the dataset into a train and test set with the train_test_split method. This method splits the dataset into two subsets: one for training the model and one for testing or evaluating its performance.
```
books = books["train"].train_test_split(test_size=0.2)
```
Example:
```
books["train"][0]
```
{'id': '21075',
 'translation': {'en': '“Why, Friday,” says I, “do you think they are going to eat them, then?” “Yes,” says Friday, “they will eat them.”',
  'fr': "Je pus distinguer que l'un de ces trois faisait les gestes les plus passionnés, des gestes d'imploration, de douleur et de désespoir, allant jusqu'à une sorte d'extravagance."}}
















