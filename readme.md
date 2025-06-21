<h1 align="center">   🧠 Automated brain explanations 🧠</h1>
<p align="center">
<img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9--3.11-blue">
  <img src="https://img.shields.io/badge/numpy->=2.0-blue">
</p>  

**How does the brain process language?** We've been studying this question using large-scale brain-imaging datasets collected from human subjects as they read and listen to stories.
Along the way, we've used LLMs to help us predict and explain patterns in this data and found a bunch of cool things!
This repo contains code for doing these analyses & applying the tools we've developed to various domains.

### Reference
**This repo contains code underlying 2 neuroscience studies:**

<details>
<summary>Generative causal testing to bridge data-driven models and scientific theories in language neuroscience (<a href="https://arxiv.org/abs/2410.00812">Antonello*, Singh*, et al., 2024, arXiv</a>)
</summary>
<br>
Representations from large language models are highly effective at predicting BOLD fMRI responses to language stimuli. However, these representations are largely opaque: it is unclear what features of the language stimulus drive the response in each brain area. We present generative causal testing (GCT), a framework for generating concise explanations of language selectivity in the brain from predictive models and then testing those explanations in follow-up experiments using LLM-generated stimuli. This approach is successful at explaining selectivity both in individual voxels and cortical regions of interest (ROIs), including newly identified microROIs in prefrontal cortex. We show that explanatory accuracy is closely related to the predictive power and stability of the underlying predictive models. Finally, we show that GCT can dissect fine-grained differences between brain areas with similar functional selectivity. These results demonstrate that LLMs can be used to bridge the widening gap between data-driven models and formal scientific theories.
</details>
<details>
<summary>Evaluating scientific theories as predictive models in language neuroscience (Singh*, Antonello*, et al. 2024, in prep)
</summary>
<br>
Modern data-driven encoding models are highly effective at predicting brain responses
to language stimuli. However, these models struggle to explain the underlying phenomena,
i.e. what features of the stimulus drive the response in each brain area? We present Question Answering encoding models, a method for converting qualitative theories of language selectivity in the brain into highly accurate, interpretable models of brain responses. QA encoding models annotate a language stimulus by using a large language model to answer yes-no questions corresponding to qualitative theories. A compact QA encoding model that uses only 35 questions outperforms existing baselines at predicting brain responses to language stimuli in both fMRI and ECoG data. The model weights also provide easily interpretable maps of language selectivity across cortex. We find that these selectivity maps quantitatively match meta-analyses of the existing literature. We further evaluate these selectivity maps in a follow-up fMRI experiment and find strong agreement between the maps and responses to synthetic stimuli designed to test their selectivity. These results demonstrate that LLMs can bridge the widening gap between qualitative scientific theories and data-driven models.
</details>
<br>

**This repo also contains code for experiments in 3 ML papers** (for a simple scikit-learn interface to use these, see [imodelsX](https://github.com/csinva/imodelsX)):
<details>
<summary>Augmenting interpretable models with large language models during training (<a href="https://www.nature.com/articles/s41467-023-43713-1">Singh et al. 2023, Nature communications</a>)
</summary>
<br>
Recent large language models (LLMs), such as ChatGPT, have demonstrated remarkable prediction performance for a growing array of tasks. However, their proliferation into high-stakes domains and compute-limited settings has created a burgeoning need for interpretability and efficiency. We address this need by proposing Aug-imodels, a framework for leveraging the knowledge learned by LLMs to build extremely efficient and interpretable prediction models. Aug-imodels use LLMs during fitting but not during inference, allowing complete transparency and often a speed/memory improvement of greater than 1000x for inference compared to LLMs. We explore two instantiations of Aug-imodels in natural-language processing: Aug-Linear, which augments a linear model with decoupled embeddings from an LLM and Aug-Tree, which augments a decision tree with LLM feature expansions. Across a variety of text-classification datasets, both outperform their non-augmented, interpretable counterparts. Aug-Linear can even outperform much larger models, e.g. a 6-billion parameter GPT-J model, despite having 10,000x fewer parameters and being fully transparent. We further explore Aug-imodels in a natural-language fMRI study, where they generate interesting interpretations from scientific data.
</details>
<details>
<summary>QA-Emb: Crafting interpretable Embeddings by asking LLMs questions (<a href="https://arxiv.org/abs/2405.16714">Benara*, Singh* et al. 2024, NeurIPS</a>)
</summary>
<br>
Large language models (LLMs) have rapidly improved text embeddings for a growing array of natural-language processing tasks. However, their opaqueness and proliferation into scientific domains such as neuroscience have created a growing need for interpretability. Here, we ask whether we can obtain interpretable embeddings through LLM prompting. We introduce question-answering embeddings (QA-Emb), embeddings where each feature represents an answer to a yes/no question asked to an LLM. Training QA-Emb reduces to selecting a set of underlying questions rather than learning model weights.<br>
We use QA-Emb to flexibly generate interpretable models for predicting fMRI voxel responses to language stimuli. QA-Emb significantly outperforms an established interpretable baseline, and does so while requiring very few questions. This paves the way towards building flexible feature spaces that can concretize and evaluate our understanding of semantic brain representations. We additionally find that QA-Emb can be effectively approximated with an efficient model, and we explore broader applications in simple NLP tasks.
</details>
<details>
<summary>SASC: Explaining black box text modules in natural language with language models (<a href="https://arxiv.org/abs/2305.09863">Singh*, Hsu*, et al. 2023, NeurIPS workshop</a>)
</summary>
<br>
SASC takes in a text module and produces a natural explanation for it that describes what it types of inputs elicit the largest response from the module (see Fig below). The GCT paper tests this in detail in an fMRI setting.
<br>
<img src="https://microsoft.github.io/automated-brain-explanations/fig.svg?sanitize=True&kill_cache=1" width="90%">

SASC is similar to the nice [concurrent paper](https://github.com/openai/automated-interpretability) by OpenAI, but simplifies explanations to describe the function rather than produce token-level activations. This makes it simpler/faster, and makes it more effective at describing semantic functions from limited data (e.g. fMRI voxels) but worse at finding patterns that depend on sequences / ordering.

To use with <a href="https://github.com/csinva/imodelsX">imodelsX</a>, install with `pip install imodelsx` then the below shows a quickstart example.

```python
from imodelsx import explain_module_sasc
# a toy module that responds to the length of a string
mod = lambda str_list: np.array([len(s) for s in str_list])

# a toy dataset where the longest strings are animals
text_str_list = ["red", "blue", "x", "1", "2", "hippopotamus", "elephant", "rhinoceros"]
explanation_dict = explain_module_sasc(
    text_str_list,
    mod,
    ngrams=1,
)
```
</details>
<br>

### Setting up

**Dataset**
- The `data/decoding` folder contains a quickstart easy example for TR-level decoding
  - it has everything needed, but if you want to visualize the results on a flatmap, you need to download the relevant PCs from [here](https://utexas.box.com/s/7ur0fsr52nephxp96hs5dxm99rk2v1u0)
- to quickstart, just download the responses / wordsequences for 3 subjects from the [encoding scaling laws paper](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915)
  - this is all the data you need if you only want to analyze 3 subjects and don't want to make flatmaps
- to run Eng1000, need to grab `em_data` directory from [here](https://github.com/HuthLab/deep-fMRI-dataset) and move its contents to `{root_dir}/em_data`
- for more, download data with `python experiments/00_load_dataset.py`
    - create a `data` dir under wherever you run it and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637)
- to make flatmaps, need to set [pycortex filestore](https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html) to `{root_dir}/ds003020/derivative/pycortex-db/`

**Code**
- `pip install -e .` from the repo directory to locally install the `neuro` package
- set `neuro.config.root_dir/data` to where you put all the data
  - loading responses
    - `neuro.data.response_utils` function `load_response`
    - loads responses from at `{neuro.config.root_dir}/ds003020/derivative/preprocessed_data/{subject}`, where they are stored in an h5 file for each story, e.g. `wheretheressmoke.h5`
  - loading stimulus
    - `ridge_utils.features.stim_utils` function `load_story_wordseqs`
    - loads textgrids from `{root_dir}/ds003020/derivative/TextGrids`, where each story has a TextGrid file, e.g. `wheretheressmoke.TextGrid`
    - uses `{root_dir}/ds003020/derivative/respdict.json` to get the length of each story

**Demo**
- `python experiments/02_fit_encoding.py`
    - This script takes many relevant arguments through argparse

### Reference
- see related [fMRI experiments](https://github.com/csinva/fmri)
- built from [this template](https://github.com/csinva/cookiecutter-ml-research)
