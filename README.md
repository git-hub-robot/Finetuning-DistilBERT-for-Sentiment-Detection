# Finetuning-DistilBERT-for-Sentiment-Detection
A natural language processing project that demonstrates parameter efficient fine-tuning (PEFT) using Low Rank Adaptation (LoRA) on distilbert-base-uncased. The task type is sequence classification and uses the SST-2 dataset.
## Overview
- Objective: Finetune a pretrained Transformer model efficiently for sentiment detection.
- Methodology: Use PEFT (LoRA) targetting only query ('q_lin') and value ('v_lin') layers, to minimize trainable parameters.
- Dataset: 'Setfit/sst2' loaded from the HuggingFace library.
- Evaluation: Tracked training and validation loss using HuggingFace 'Trainer' and generated prediction outputs ('classification_results.csv') comparing predicted labels against true labels.
## Tools and Libraries
- Python
- Pytorch
- Hugging Face Libraries: transformers, datasets, peft, trl
- Data Handling Libraries: numpy, pandas
## Requirements
- Requires a HuggingFace token ("HF_TOKEN").
## Results
- Training Loss: 0.419
- Validation Loss: 0.346
