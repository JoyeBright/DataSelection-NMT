# DataSelection-NMT
Selecting Parallel In-domain Sentences for Neural Machine Translation Using Monolingual Texts
# How to use
We ported the best checkpoints of trained models to the Hugging Face (HF) repository. <br>
Since our models were trained with [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), it was not possible to employ them directly for inference on HF. <br>
However, we use [CTranslate2](https://github.com/OpenNMT/CTranslate2) an inference engine for transformer models to bypass this issue. <br>

