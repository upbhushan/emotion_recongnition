# Multimodal Emotion Recognition using Text, Audio and Video

This project implements a multimodal emotion recognition system that combines text, audio, and video to classify emotions in real conversational settings. The model is trained on the MELD dataset and uses the following pretrained models:
- BERT for text
- Wav2Vec2.0 for speech audio
- Vision Transformer (ViT) for visual frames

I apply Cross-Modal Attention Fusion and Temporal Video Encoding (ViT + BiLSTM + Attention) to learn meaningful interactions between modalities. The final system achieves a Weighted F1-score of 77.65%, outperforming unimodal and basic fusion baselines.

---

## Features
- Multimodal processing: Text + Audio + Video
- Uniform frame extraction from videos
- Temporal video modeling using BiLSTM + Attention
- Cross-modal attention-based fusion
- Silent audio detection and masking
- Precomputed embeddings for fast training
- Unimodal and bimodal ablation studies
- Evaluation using Accuracy, Weighted F1, Macro F1
- Support for attention visualization

---

## Model Architecture Overview

Text (BERT):
- CLS token embedding used as sentence representation

Audio (Wav2Vec2):
- Raw waveform converted into deep speech embeddings
- Silent audio automatically masked

Video (ViT + BiLSTM + Attention):
- Uniformly sample 8 frames from each video
- ViT extracts embeddings for each frame
- BiLSTM learns temporal changes in expressions
- Multi-head attention selects important frames

Fusion (Cross-Modal Attention):
- Text attends to Audio & Video
- Audio attends to Text & Video
- Video attends to Text & Audio
- Produces fused embeddings for final classification

Classifier:
- Dense layers + dropout
- Softmax output over 7 emotion classes:
  Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

---

## Results (Weighted F1)

Text Only: 0.666  
Audio Only: 0.356  
Video Only: 0.252  
Text + Audio (Best): 0.776  
Text + Video: 0.529  
Audio + Video: 0.305  
Trimodal (Text + Audio + Video): 0.576  

Best-performing combination: Text + Audio  
Final best model: Weighted F1-score = 77.65%

---

## Dataset

I use the MELD (Multimodal EmotionLines Dataset), containing:
- 13,000+ utterances
- Multi-party dialogues from the show “Friends”
- Aligned text, audio, and video files
- 7 emotion categories

Dataset link:
https://affective-meld.github.io/

---

## Installation

```bash
pip install -r requirements.txt
