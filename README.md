# Transformer Model from Scratch with PyTorch

This project demonstrates a full implementation of the **Transformer architecture** using PyTorch, from basic components like Multi-Head Attention and Positional Encoding to a complete encoder-decoder model suitable for sequence-to-sequence tasks (e.g., machine translation).

---

## 🚀 Features

- Custom implementation of:
  - Multi-Head Self Attention
  - Position-wise Feed-Forward Networks
  - Positional Encoding
  - Encoder & Decoder Layers
- Full Transformer model
- Training loop with synthetic data
- Loss decreases consistently across epochs

---

## 🧱 Architecture Overview

This implementation follows the original Transformer architecture as introduced in the ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper:

- Input Embedding + Positional Encoding
- Stacked Encoder and Decoder layers
- Masking for source and target sequences
- Final linear + softmax projection

---

## 📦 Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.7

Install dependencies:
```bash
pip install torch
```

---

## 🧪 Training

```python
# Initialize model
transformer = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
)

# Train on random data
for epoch in range(100):
    ...
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
```

Loss reduces steadily over 100 epochs:

```
Epoch: 1, Loss: 8.67
...
Epoch: 50, Loss: 5.15
...
Epoch: 100, Loss: 2.75
```

---

## 📊 Sample Output

Model trained on synthetic data, demonstrating convergence and functionality of core Transformer components. You can easily adapt this code to work with real datasets like WMT, IWSLT, or custom corpora.

---

## 📁 File Structure

```
├── multihead_attention.py   # MultiHeadAttention module
├── encoder_layer.py         # EncoderLayer class
├── decoder_layer.py         # DecoderLayer class
├── transformer.py           # Full Transformer architecture
├── train.py                 # Training loop and data generation
```

---

## 📚 Future Work

- Integrate real datasets (e.g., translation, summarization)
- Add beam search decoding
- Implement label smoothing
- Fine-tune on pre-trained embeddings (e.g., GloVe, FastText)

---

## Credits

Inspired by the seminal paper:  
[**Attention is All You Need** (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## License

This project is licensed under the MIT License.
