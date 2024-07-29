<div>
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120">
  <!-- Definitions for gradient -->
  <defs>
    <radialGradient id="grad1" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" style="stop-color:#ff8a8a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ff0000;stop-opacity:1" />
    </radialGradient>
  </defs>
  <!-- Outer circle with gradient fill -->
  <circle cx="60" cy="60" r="50" fill="url(#grad1)" />
  <!-- Middle circle -->
  <circle cx="60" cy="60" r="35" fill="none" stroke="#ffffff" stroke-width="4"/>
  <!-- Inner circle -->
  <circle cx="60" cy="60" r="20" fill="none" stroke="#ffffff" stroke-width="4"/>
  <!-- Decorative element: a smaller circle offset from the center -->
  <circle cx="80" cy="40" r="5" fill="#ffffff"/>
</svg>
</div>

# Omni Embedding Service (OES)

OES is a self-hostable embeddings service. It allows you to embed data of various types (text, image, audio, etc.) for applications such as RAG, search, model training, etc.

# Quick Start

Create a `config.json` file like so:

```yaml
---
models:
- model_name: openai/clip-vit-base-patch32
  encodings:
  - data_type: text
    replicas: 1
  - data_type: image
    replicas: 1
```

```bash
cargo run -- run
```

Now you can embed data using the API:

```python
import base64
import openai
import requests
from PIL import Image
from io import BytesIO
import numpy as np

client = openai.Client(api_key="sk", base_url="http://localhost:8080/oai/")

text_embedding1 = client.embeddings.create(
    model="openai/clip-vit-base-patch32/text",
    input="a cat"
)
text_embedding2 = client.embeddings.create(
    model="openai/clip-vit-base-patch32/text",
    input="a yummy potato"
)

def image_to_dataurl(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

image_url = "https://www.cats.org.uk/uploads/images/featurebox_sidebar_kids/Cat-Behaviour.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
image_embedding = client.embeddings.create(
    model="openai/clip-vit-base-patch32/image",
    input=image_to_dataurl(image)
)

emb1 = np.array(text_embedding1.data[0].embedding)
emb2 = np.array(text_embedding2.data[0].embedding)
emb3 = np.array(image_embedding.data[0].embedding)
print(f"Similarity between 'a cat' and image: {np.dot(emb1, emb3)}")
print(f"Similarity between 'a potato' and image: {np.dot(emb2, emb3)}")
```
