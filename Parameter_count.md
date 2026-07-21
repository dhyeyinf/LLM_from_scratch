# GPT-2 Parameter Count Breakdown

## Embedding Parameters
- Token embeddings: `50257 * 768 = 38.4M`
- Positional embeddings: `50257 * 768 = 38.4M`

> **Total embedding parameters = 38.4M**

---

## Transformer Block

### a) Multi-Head Attention
1. Key, Query, Value weights: `3 * 768 * 768 = 1.77M`
2. Output projection: `768 * 768 = 0.59M`
3. **Total = 2.36M**

### b) Feed Forward Neural Network
- Expansion layer: `768 * (4 * 768) = 2.36M`
- Contraction layer: `(4 * 768) * 768 = 2.36M`

> **Total FFN = 4.72M**

### Per-Block Total
`4.72M + 2.36M = 7.08M`

### All 12 Transformer Blocks
`12 * 7.08M = 85.2M`

---

## Final Layer (Softmax Output)
`768 * 50257 = 38.4M`

---

## Total Parameter Count

| Component | Parameters |
|---|---|
| Embedding layer | 38.4M |
| Transformer blocks (×12) | 85.2M |
| Final output layer | 38.4M |
| **Total** | **162M** |
