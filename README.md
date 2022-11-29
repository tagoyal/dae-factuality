# dae-factuality

**UPDATE: Models in this repo measure factuality for single sentence source sentences. For longer source docuemnts (e.g. XSum/CnnDm articles), please refer to our follow-up work https://github.com/tagoyal/factuality-datasets**

Code for paper "Evaluating Factuality in Generation with Dependency-level Entailment" https://arxiv.org/pdf/2010.05478.pdf

```
@inproceedings{goyal2020evaluating,
  title={Evaluating Factuality in Generation with Dependency-level Entailment},
  author={Goyal, Tanya and Durrett, Greg},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2020},
  year={2020}
}
```


# Dependency Arc Entailment (DAE)

The DAE formulation decomposes the summary-level factuality task in smaller entailment tasks at the arc-level. Given the input article, the DAE model is trained to independantly predict whether the relationship implied by each independant dependency arc is entailed by the input or not. 

<img width="382" alt="Screen Shot 2022-11-29 at 1 49 32 AM" src="https://user-images.githubusercontent.com/22390810/204469849-dd016288-4920-4363-801f-543a15ac8261.png">

In the above example, the arc woman -> Chicago is non-factual because the woman is travelling to Chicago but not necessarily from Chicago. All other arcs in the summary are factual according to the DAE formulation

# Models and Data

Environment base is Python 3.6. Also see requirements.txt. We used Stanford CoreNLP version 3.9.1.

They can be downloaded from here: https://drive.google.com/drive/folders/16NEL8T-JvhJPy7miVUbMELVE8ZOTYGit?usp=sharing

There are 3 models in the above folder:

dae: model trained on only the paraphrase data, without synonym augmentation or hallucinations

dae_w_syn (this model used in experimts in Section 6 onwards): trained on paraphrase data + synonym data

dae_w_syn_w_hallu: trained on paraphrase data + synonym data + hallucination data

# Evaluation
To evaluate the model on your own data (or that in resources folder), run the evaluate_factuality script. This relies on the stanford CoreNLP server to parse the output and obtain it's dependency parse. 

1) Run the coreNLP server. We used Stanford CoreNLP version 3.9.1 (can be downloaded from https://stanfordnlp.github.io/CoreNLP/history.html). Run the following command (from the stanford corenlp parser folder): 
```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
```
2) Next, run the evaluation scipt with the appropriate pointers to model directory and input: 
```
python evaluate_factuality.py --model_type electra_dae --input_dir [model_dir] --test_type [summ/para]
```
