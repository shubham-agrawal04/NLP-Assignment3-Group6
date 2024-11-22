# NLP Assignment 3: Fine-Tuning Llama3.2-1B Model  
**Group 6**  

## Project Overview  
This repository contains the code and results for NLP Assignment 3, where we fine-tuned the **Llama3.2-1B** model for two tasks:  
1. **Text Classification (SST-2 dataset)**  
2. **Question-Answering (SQuAD dataset)**  

Key objectives include:  
- Calculating and verifying the number of model parameters.  
- Evaluating performance on test splits for pre-trained (zero-shot) and fine-tuned models using task-specific metrics.  
- Understanding the impact of fine-tuning on model parameters and performance.  

---

## Tasks and Metrics  
### 1. **Classification: SST-2**  
- Dataset: Stanford Sentiment Treebank (SST-2)  
- Metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

### 2. **Question-Answering: SQuAD**  
- Dataset: Stanford Question Answering Dataset (SQuAD)  
- Metrics:  
  - squad_v2  
  - F1 Score  
  - METEOR  
  - BLEU  
  - ROUGE  
  - Exact Match  

---

## Methodology  
1. **Model Selection**:  
   - Pre-trained **Llama3.2-1B** model from 🤗 Transformers library.  

2. **Train-Test Split**:  
   - 80:20 split with `random`/`stratify` sampling.  
   - Seed: `1`.  

3. **Fine-Tuning**:  
   - **Classification Task**: `AutoModelForSequenceClassification`.  
   - **Question-Answering Task**: `AutoModelForQuestionAnswering`.  
   - Training performed on train splits, and evaluation metrics computed on test splits.  

4. **Metrics Calculation**:  
   - For the zero-shot (pre-trained) and fine-tuned models.  

5. **Model Parameters**:  
   - Parameter count verified pre- and post-fine-tuning.  

6. **Model Deployment**:  
   - Fine-tuned model pushed to 🤗 Hub.  

---

## Files Description

| File Description                                           | Link                                                                 |
|------------------------------------------------------------|----------------------------------------------------------------------|
| **Base Model Parameter Calculation**: Code for calculating and verifying the parameters of the pre-trained model. | [Link to File](./nlp-assignment-3-group-6-basemodel.ipynb)                                                   |
| **SST-2 Classification Model**: Fine-tuning and evaluation script for the SST-2 classification task.            | [Link to File](./nlp-assignment-3-group-6-classification%20(1).ipynb)                                                   |
| **SQuAD Question-Answering Model (4,000 data points)**: Fine-tuning and evaluation script with 4,000 training and 1,000 testing data points. | [Link to File](./nlp-assignment-3-group-6-qa%20(1).ipynb)                                                   |
| **SQuAD Question-Answering Model (24,000 data points)**: Fine-tuning and evaluation script with 24,000 training and 1,000 testing data points. | [Link to File](./nlp-assignment-3-group-6-qa%20(2).ipynb)                                                   |
| **Assignment Report**: Results and Observation for the Assignment. | [Link to File]()                     |

## Contributors  
- **Group 6 Members**:  
  - Member 1: Shubham Agrawal (22110249)
  - Member 2: Pratham Sharda (22110203)
  - Member 3: Harshit (22110095)
  - Member 4: Chirag Patel (22110183)
  - Member 5: Pranjal Gaur (22110201)
  - Member 6: Nimitt (22110169)

---

## Acknowledgements  
- **Hugging Face Transformers** for the pre-trained Llama3.2-1B model.  
- Datasets: **SST-2**, **SQuAD**.  

