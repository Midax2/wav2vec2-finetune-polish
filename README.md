# wav2vec2-finetune-polish

Fine-tune Facebook’s Wav2Vec 2.0 English automatic speech recognition (ASR) model on Polish speech data, with a custom Polish tokenizer and export to TensorFlow `.h5` format.

## Overview

This project provides a step-by-step pipeline to:

- Prepare Polish speech datasets
- Create a tokenizer supporting Polish characters
- Fine-tune the English Wav2Vec 2.0 model on Polish audio and transcripts
- Export the trained model to `.h5` format compatible with TensorFlow/Keras

## Requirements

- Python 3.7+
- `transformers`
- `datasets`
- `torchaudio`
- `onnx`, `onnxruntime`, `onnx-tf`
- `tensorflow`
- `librosa`
- `jiwer`

Install dependencies with:

```bash
pip install transformers datasets torchaudio librosa jiwer onnx onnxruntime onnx-tf tf2onnx tensorflow
```

## Usage
1. Prepare your Polish audio dataset in WAV format (16kHz sample rate).

2. Edit the data variable in the script to add your audio file paths and corresponding Polish transcripts.

3. Run the fine-tuning script:

```bash
python full_finetune_wav2vec2_polish.py
```

4. The fine-tuned model will be saved as final_model_polish.h5 ready for TensorFlow/Keras use.

## Files

- `full_finetune_wav2vec2_polish.py`: Complete training and export script.

- `vocab.json`: Polish tokenizer vocabulary file (auto-generated).

- `final_model_polish.h5`: Exported TensorFlow model (after training).

## Notes

- Make sure your WAV audio files are sampled at 16kHz.

- Training time depends on dataset size and hardware.

- This script uses Hugging Face Transformers and ONNX for conversion.

## License
> MIT License
