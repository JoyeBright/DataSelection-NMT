# DataSelection-NMT
Selecting Parallel In-domain Sentences for Neural Machine Translation Using Monolingual Texts

# Our Pre-trained models on Hugging Face
|Systems        | Link | Systems | Link | 
|:-------------:|:----:|:-------:|:----:|
|Top1         |[Download](https://huggingface.co/joyebright/Top1-with-without-mixing/resolve/main/Top1-withBPE-step-5000.pt)|Top1|[Download](https://huggingface.co/joyebright/Top1-with-without-mixing/resolve/main/Top1-withBPE-step-5000.pt)|  
|Top2+Top1    |[Download](https://huggingface.co/joyebright/Top2-with-mixing/resolve/main/top2-withBPE-step-10000.pt)|Top2|[Download](https://huggingface.co/joyebright/Top2-without-mixing/resolve/main/top2-withBPE-step-5000.pt)|
|Top3+Top2+...|[Download](https://huggingface.co/joyebright/Top3-with-mixing/resolve/main/top3-withBPE-step-13000.pt)|Top3|[Donwload](https://huggingface.co/joyebright/Top3-without-mixing/resolve/main/top3-withBPE-step-5000.pt)|
|Top4+Top3+...|[Download](https://huggingface.co/joyebright/Top4-with-mixing/resolve/main/top4-withBPE-step-17000.pt)|Top4|[Donwload](https://huggingface.co/joyebright/Top4-without-mixing/resolve/main/top4-withBPE-step-5000.pt)|
|Top5+Top4+...|[Download](https://huggingface.co/joyebright/Top5-with-mixing/resolve/main/top5-withBPE-step-20000.pt)|Top5|[Donwload](https://huggingface.co/joyebright/Top5-without-mixing/resolve/main/top5-withBPE-step-5000.pt)|
|Top6+Top5+...|[Download](https://huggingface.co/joyebright/Top6-with-mixing/resolve/main/top6-withBPE-step-19000.pt)|Top6|[Donwload](https://huggingface.co/joyebright/Top6-without-mixing/resolve/main/top6-withBPE-step-5000.pt)|


## How to use
**Note:** we ported the best checkpoints of trained models to the Hugging Face (HF). Since our models were trained by [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), it was not possible to employ them directly for inference on HF. To bypass this issue, we use [CTranslate2](https://github.com/OpenNMT/CTranslate2)– an inference engine for transformer models.

Follow steps below to translate your sentences:

1\. **Install the Python package**:
```bash
pip install --upgrade pip
pip install ctranslate2
```
2\. **Download models from our [HF repository](# Our):**



