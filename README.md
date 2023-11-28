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

## Hugging Face tutorial: 
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

Load a T5 tokenizer to process the English-French language pairs:
```
from transformers import AutoTokenizer

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
The preprocessing function you want to create needs to:

1. Prefix the input with a prompt so T5 knows this is a translation task. Some models capable of multiple NLP tasks require prompting for specific tasks.
2. Tokenize the input (English) and target (French) separately because you can't tokenize French text with a tokenizer pretrained on an English vocabulary.
3. Truncate sequences to be no longer than the maximum length set by the max_length parameter.

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

```
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
```
To apply the preprocessing function over the entire dataset, use datasets map method. We can speed up the ```map``` function by setting ```batched=True``` to process multiple elements of the dataset at once:

```
tokenized_books = books.map(preprocess_function, batched=True)
```

Now create a batch of examples using DataCollatorForSeq2Seq. It's more efficient to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

```
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```

### Step 3: Model performance evaluation </br>
It is beneficial to assess your model's performance during training by incorporating a metric. You can efficiently integrate an evaluation method using the Evaluate library. Specifically, for this task, you should load the SacreBLEU metric. Refer to the Evaluate quick tour to understand how to load and compute a metric effectively.

```
import evaluate
metric = evaluate.load("sacrebleu")
```

Then create a function that passes your predictions and labels to compute to calculate the SacreBLEU score:

import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

```
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
```
Your ```compute_metrics``` function is ready to go now, and you'll return to it when you setup your training.

### Step 4: Training your own model </br>

####Load T5 with TFAutoModelForSeq2SeqLM:
```
from transformers import TFAutoModelForSeq2SeqLM
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

Convert your datasets to the tf.data.Dataset format with prepare_tf_dataset():
```
tf_train_set = model.prepare_tf_dataset(
    tokenized_books["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_books["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
```
Configure the model for training with compile. Note that Transformers models all have a default task-relevant loss function, so you don't need to specify one unless you want to:

```
import tensorflow as tf
model.compile(optimizer=optimizer)  # No loss argument!
```

We can start training our model! Call ```fit``` with your training and validation datasets, the number of epochs, and your callbacks to finetune the model:
```
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, "path/to/save/model")
```

### Step 5: Running inference on our trained model or on a pre-trained model </br>
For T5, you need to prefix your input depending on the task you're working on. For translation from English to French, you should prefix your input as shown below:
```
text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```
The simplest way to try out our finetuned model or pre-trained model for inference is to use it in a pipeline(). Instantiate a pipeline for translation with your model, and pass your text to it:

```
from transformers import pipeline

translator = pipeline("translation", model="my_awesome_opus_books_model")
translator(text)
```
{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}

You can also manually replicate the results of the pipeline if you'd like:

Tokenize the text and return the input_ids as TensorFlow tensors:

from transformers import AutoTokenizer
```
tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
inputs = tokenizer(text, return_tensors="tf").input_ids
```





