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

#### Step 1: Installation of necessary libraries
Installation of libraries that are often used in the development and evaluation of natural language processing models.
``` 
pip install transformers datasets evaluate sacrebleu
```
</br>
#### Step 2: Load OPUS Books dataset
Test the English-French T5 subset of the OPUS Books dataset and translate English text into French.


- Cesta k kontrolnímu bodu modelu: cbqa/small_ssm_nq/model.ckpt-1110000
- Soubor konfigurace modelu (Gin): models/t5_1_1_small.gin


