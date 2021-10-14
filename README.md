# DataSelection-NMT
Selecting Parallel In-domain Sentences for Neural Machine Translation Using Monolingual Texts

## Our Pre-trained models on Hugging Face
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
2\. **Download models from our [HF repository](#Our-Pre-trained-models-on-Hugging-Face):**
You can do this manually or use the following python script:
```python
import requests

url = "Download Link"
model_path = "Model Path"
r = requests.get(url, allow_redirects=True)
open(model_path, 'wb').write(r.content)
```
3\. **Convert the downloaded model:**
```bash
ct2-opennmt-py-converter --model_path model_path --output_dir output_directory
```
3\. **Translate tokenized inputs:**

**Note:** the inputs should be tokenized by [SentencePiece](https://github.com/google/sentencepiece). 
You can also use tokenized version of [IWSLT test sets](https://github.com/JoyeBright/DataSelection-NMT/tree/main/Data-Table1/Dev%20and%20Test%20Sets).

```python
import ctranslate2
translator = ctranslate2.Translator("output_directory/")
translator.translate_batch([["▁H", "ello", "▁world", "!"]])
```
or
```python
import ctranslate2
translator = ctranslate2.Translator("output_directory/")
translator.translate_file(input_file, output_file, batch_type= "tokens/examples")
```
To customize the CTranslate2 functions, read this [API document](https://github.com/OpenNMT/CTranslate2/blob/master/docs/python.md).

4\. **Detokenize the outputs:**

**Note:** you need to [detokenize](https://github.com/JoyeBright/DataSelection-NMT/blob/main/Tools/detokenizer.perl) the output with the same sentencepiece model as used in step 3.

```bash
tools/detokenize.perl -no-escape -l fr \
< output_file \
> output_file.detok
```



