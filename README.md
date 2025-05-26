# Evaluating Lexical Proficiency in Neural Language Models

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](img/diagram.svg)

This is the public repository for our paper: *Evaluating Lexical Proficiency in Neural Language Models*, C. Ciaccio, A. Miaschi, F. Dell'Orletta (ACL 2025). 

The repository contains the resources and code that we developed in order to run our experimets for assessing the lexical proficiency of Italian neural language models. Specifically:

## Datasets

The **data** folder contains:
- 100-neos.csv &rarr; neologism dataset
- 100-nonce.csv &rarr; nonce words dataset
- ONLI-NEO &rarr; data extracted from **[Osservatorio Neologico della Lingua Italiana, ONLI](https://www.iliesi.cnr.it/ONLI/)**.
- it-dictionary-gz &rarr; data extracted from the Italian Wiktionary **[Wikizionario](https://it.wiktionary.org/wiki/Pagina_principale)**
- (train-test-val)_dataset.csv &rarr; the splits for train, test and validation used in our experimets

More resources related on the Wiktionary data format, the parser and the ONLI scraper can be found in our **[Italian Wiktionary Parser](https://github.com/snizio/italian-wiktionary-parser)** repository.

## Code

The **code** folder contains the file "finetuningT5.py" that we used to finetune all T5 models in a text-to-text multi-task learning setup (training + evaluation). 

## Human evaluation data

The **annotation** folder contains the human annotated scores of novelty and adhesion for the nonce words setting for each model (in batches of 25). 

## 

If you use any of the following contents for your work, we kindly ask you to cite our paper:

```bibtex
@inproceedings{
}
```

> **Abstract:** We present a novel evaluation framework de-
signed to assess the lexical proficiency and
linguistic creativity of Transformer-based Lan-
guage Models (LMs). We validate the frame-
work by analyzing the performance of a set of
LMs of different sizes, in both mono- and mul-
tilingual configuration, across tasks involving
the generation, definition, and contextual usage
of lexicalized words, neologisms, and nonce
words. To support these evaluations, we devel-
oped a novel dataset of lexical entries for the
Italian language, including curated definitions
and usage examples sourced from various on-
line platforms. The results highlight the robust-
ness and effectiveness of our framework in eval-
uating multiple dimensions of LMs’ linguistic
understanding and offers an insight, through
the assessment of their linguistic creativity, on
the lexical generalization abilities of LMs.