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

#### Tento repozitář popisuje neúspěšný pokus o spuštění inference na již natrénovaném T5 modelu s datovou sadou v TensorFlow "wmt_t2t_translate".
 #### ❌ Google Colab .ipynb 
  #### ❌ Lokálně
</br>

## Postup: 
Naklonování repozitáře ```git clone https://github.com/google-research/t5x.git``` ✅

#### Krok 1: Vyberte model
Pro spuštění inference na modelu potřebujete soubor konfigurace Gin, který definuje parametry modelu a cestu k jeho kontrolnímu bodu. Pro tento příklad použijeme model T5-1.1-Small, který byl jemně naladěn na úloze natural_questions_open_test SeqIO:

- Cesta k kontrolnímu bodu modelu: cbqa/small_ssm_nq/model.ckpt-1110000
- Soubor konfigurace modelu (Gin): models/t5_1_1_small.gin

### Chyby při lokálním spuštění: 
- neúplná dokumentace
- chybějící modul - ``` t5x.infer_unfragmented ```, který slouží ke spuštění inference, došlo k jeho smazání z repozitáře

