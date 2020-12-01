# dae-factuality
Code for paper "Evaluating Factuality in Generation with Dependency-level Entailment" https://arxiv.org/pdf/2010.05478.pdf
(detailed readme will be updated soon)

# Models and Data
They can be downloaded from here: https://drive.google.com/drive/folders/1BJWpmRM-4qcn0vaFPo0k_xO5s-N9BMfS?usp=sharing

There are 3 models in the above folder:
dae: model trained on only the paraphrase data, without synonym augmentation or hallucinations
dae_w_syn (this model used in experimts in Section 6 onwards): trained on paraphrase data + synonym data
dae_w_syn_w_hallu: trained on paraphrase data + synonym data + hallucination data

# Evaluation
To evaluate the model on your own data, run the evaluate_factuality.py script. 
