# Task-01: Text Generation with GPT-2 🍭🌱  
**ProDigy Infotech**

## Overview  
This project focuses on **text generation** using **GPT-2**, a transformer model developed by OpenAI. The goal is to fine-tune GPT-2 on a custom dataset to generate coherent and contextually relevant text based on a given prompt. By the end of this task, you will have a model that mimics the style and structure of your training data, enabling it to produce creative and consistent outputs.

### Key Learning Objectives:
1. Fine-tuning pre-trained language models (GPT-2).
2. Understanding text generation techniques such as **greedy search**, **beam search**, **sampling**, **top-k sampling**, and **top-p sampling**.
3. Exploring hyperparameter tuning for optimal model performance.
4. Evaluating the generated text for coherence and relevance.

---

## Steps

### 1. **Set Up the Environment**  
   Install required libraries:
   ```bash
   pip install transformers torch datasets
   ```

### 2. **Model and Tokenizer Setup**  
   Load the pre-trained GPT-2 model and tokenizer:
   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer

   model = GPT2LMHeadModel.from_pretrained('gpt2')
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   ```

### 3. **Prepare Training Data**  
   - Obtain or create a custom dataset relevant to your task.
   - Tokenize the dataset using GPT-2's tokenizer.
   ```python
   def tokenize_data(texts):
       return tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
   ```

### 4. **Fine-Tune the Model**  
   Use the **Transformers** library to fine-tune GPT-2 on your dataset:
   ```python
   from transformers import AdamW

   model.train()
   optimizer = AdamW(model.parameters(), lr=5e-5)

   for epoch in range(3):
       for batch in data_loader:
           outputs = model(**batch, labels=batch['input_ids'])
           loss = outputs.loss
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
   ```

### 5. **Text Generation Techniques**  
   Experiment with different decoding strategies:
   - **Greedy Search**:
   ```python
   output = model.generate(input_ids=tokenized_prompt, max_length=50)
   ```
   - **Beam Search**, **Sampling**, **Top-K**, **Top-P** sampling are also available.

### 6. **Evaluation and Refinement**  
   - Evaluate the model based on perplexity scores and manual assessment.
   - Adjust the training process and hyperparameters for better results.

### 7. **Deployment**  
   Save the fine-tuned model:
   ```python
   model.save_pretrained('./fine_tuned_gpt2')
   tokenizer.save_pretrained('./fine_tuned_gpt2')
   ```

---

## References  
- [How to Generate Text](https://huggingface.co/blog/how-to-generate)  
- [Colab Example](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)

---

## Project Code  
The project code is available in the repository under the folder `PRODIGY_GA_01`.

---

**Note for Generative AI Interns**:  
While output is important, understanding the **theory** behind these complex topics is key. Focus on learning the underlying concepts to ensure that you grasp how models like GPT-2 work at their core.
