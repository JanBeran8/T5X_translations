# T5X_translations
The project's primary objective is to replicate the capabilities of T5X, a cutting-edge framework designed for high-performance sequence model tasks, with a specific emphasis on language translation. 

### Časová osa:
- [x] přiřazení osob, linky na github, data vystoupení (17.10.2023)
- [ ] poslední vystoupení (11.12.2023)
- [ ] náhradní termíny (3.1.,8.1.)
- [ ] udělování KZ (11.1)

#### Tento repozitář popisuje neúspěšný pokus o zprovoznění projektu, který pomocí natrénovaného modelu na AI dokáže překládat datasets.
 ## </br> ❌ Test v Google Colab .ipynb 
  ##    </br> ❌ Test lokálně

### Google Colab:
"s lokálním spuštěním problémy s XManager, problémy s připojením na Google Cloud </br>"
  </br> ✅ Autentizace v Google Colab - bez problémů
 </br>  ✅ Naklonování repozitářů https://github.com/google-research/t5x a https://github.com/google-research/text-to-text-transfer-transformer - bez problémů
 </br>  ✅ Instalace knihoven z repozitářů - bylo potřeba nainstalovat jiné verze -> requirements.txt
 </br>  ✅ Založení účtu na Google Cloud (Console) -> vytvoření servisního účtu -> vygenerování SERVICE ACCOUNT KEY - cesta ke klíči: **gs://test_bucket-1128/my-project-1-401520-40f92252df59.json**
 </br>  ✅ Instalace knihovny Google-Cloud Storage (GCS)
 </br>  ✅ Přihlášení v Google Colab ke Google Cloud Storage (GCS) pomocí service account key</br>
           ```
            storage_client = storage.Client.from_service_account_json('my-project-1-401520-40f92252df59.json')
          ```
  </br> ✅ Import knihoven TensorFlow pro strojové učení</br>
    ```
        import tensorflow as tf
     ```
    </br> 
     ```
        import tensorflow_datasets as tfds
    ```
</br> ✅ Přístup ke Google Cloud Storage </br>
          ```
          Running on TPU: grpc://10.45.163.2:8470
          ```
          </br>
          ```
          env: TPU_ADDRESS=grpc://10.45.163.2:8470
          ```
</br> ✅ Import datasetu ***wmt_t2t_translate/de-en*** </br>
          ```
          dataset_name = "wmt_t2t_translate"
          ```
          </br>
          ```
          dataset, info = tfds.load(dataset_name, with_info=True, data_dir="/drive/MyDrive/dataset/wmt_t2t_translate/de-en")
          ```
          </br>
  </br> ❌
 </br> ❌
  </br> ❌
   </br> ❌
    </br> ❌
     </br> ❌
      </br> ❌






# T5X
Link: https://github.com/google-research/t5x </br>
</br>
T5X is a modular, composable, research-friendly framework for high-performance, configurable, self-service training, evaluation, and inference of sequence models (starting with language) at many scales.

It is essentially a new and improved implementation of the T5 codebase (based on Mesh TensorFlow) in JAX and Flax. To learn more, see the T5X Paper.

Below is a quick start guide for training models with TPUs on Google Cloud. For additional tutorials and background, see the complete documentation.

### Progresses
- [ ] 1.1 Installing XManager  
- [ ] 1.2 Google Cloud Platform (GCP)  
