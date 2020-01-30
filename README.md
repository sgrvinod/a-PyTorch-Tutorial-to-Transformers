This is a **[PyTorch](https://pytorch.org) Tutorial to Machine Translation**.

This is the sixth in [a series of tutorials](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) I'm writing about _implementing_ cool models on your own with the amazing PyTorch library.

Basic knowledge of PyTorch is assumed.

If you're new to PyTorch, first read [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

Questions, suggestions, or corrections can be posted as issues.

I'm using `PyTorch 1.4` in `Python 3.6`.

---

**27 Jan 2020**: Code is now available for [a PyTorch Tutorial to Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution).

---

# Contents

[***Objective***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#objective)

[***Concepts***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

[***Overview***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

[***Implementation***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

[***Training***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

[***Inference***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

[***Frequently Asked Questions***](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

# Objective

**To build a model that can translate from one language to another.**


>Um ein Modell zu erstellen, das von einer Sprache in eine andere übersetzen kann.


We will be implementing the pioneering research paper [_'Attention Is All You Need'_](https://arxiv.org/abs/1706.03762), which introduced the Transformer network to the world. A watershed moment for cutting-edge Natural Language Processing.

>Wir werden das wegweisende Forschungspapier [_"Attention Is All You Need"_](https://arxiv.org/abs/1706.03762) umsetzen, das das Transformer-Netzwerk in die Welt eingeführt hat. Ein Wendepunkt für die hochmoderne Natural Language Processing.

Specifically, we are going to be translating from **English** to **German**. And yes, everything written here in German is straight from the horse's mouth! (The horse, of course, being the model.)

>Konkret werden wir vom **Englischen** ins **Deutsche** übersetzen. Und ja, alles, was hier in deutscher Sprache geschrieben wird, ist direkt aus dem Mund des Pferdes! (Das Pferd ist natürlich das Modell.)


# Tutorial in Progress

I am still writing this tutorial.

<p align="center">
<img src="./img/incomplete.jpg">
</p>

In the meantime, **you could take a look at the code** – it works!

The trained model checkpoint is available [here](https://drive.google.com/drive/folders/18ltkGJ2P_cV-0AyMrbojN0Ig4JgYp9al?usp=sharing). You can use it directly with [`translate.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/blob/master/translate.py).

Here's how this model fares against the test set, as calculated in [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/blob/master/eval.py):

|BLEU|Tokenization|Cased|sacreBLEU signature|
|:---:|:---:|:---:|:---:|
|**25.1**|13a|Yes|`BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3`|
|**25.6**|13a|No|`BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3`|
|**25.9**|International|Yes|`BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.4.3`|
|**26.3**|International|No|`BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.4.3`|

The first value (13a tokenization, cased) is how the BLEU score is officially calculated by [WMT](https://www.statmt.org/wmt14/translation-task.html) (`mteval-v13a.pl`).

The BLEU score reported in the paper is **27.3**. This is possibly not calculated in the same manner, however. See [these](https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-377580270) [comments](https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191) on the official repository. With the method stated there (i.e. using `get_ende_bleu.sh` and a tweaked reference), the trained model scores **26.49**.
