# 🖼️ Image Caption Generator

This project turns any image into a meaningful sentence using a vision-to-text model.  
Built just for fun, learning, and testing how well transformers can describe visuals.

---

## 🚀 What it does

Give it an image URL → it’ll download it, preprocess it, and let the transformer model do its thing.  
The output? A natural-sounding caption that describes what’s in the image.

---

## 🛠️ Tools Used

- `ViT-GPT2` model from HuggingFace (`nlpconnect/vit-gpt2-image-captioning`)
- PyTorch and TorchVision for image handling
- PIL + Matplotlib for basic image work
- Transformers from HuggingFace

---

## ⚙️ How to Run It

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/image-caption-generator.git
cd image-caption-generator
