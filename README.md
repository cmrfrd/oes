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

Clone the repository and build the project:

```bash
git clone https://github.com/cmrfrd/oes.git
docker build -t oes -f .docker/Dockerfile .
```

Create an OES `config.yaml`. Example:

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

Now run oes with the config

```bash
docker run \
    --rm \
    --name oes \
    -p 8080:8080 \
    -v ./config.yaml:/config.yaml:Z \
    -it oes run --model-config config.yaml
```

# Using OES via the `openai` Python client

OES is compatible with the OpenAI Python client. You can use the OpenAI Python client to interact with OES.

```python
import base64
import openai
import requests
from PIL import Image
from io import BytesIO
import numpy as np

client = openai.Client(api_key="n/a", base_url="http://localhost:8080/oai/")

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

Text embeddings

```python
import base64
import openai
import requests
from PIL import Image
from io import BytesIO
import numpy as np

client = openai.Client(api_key="n/a", base_url="http://localhost:8080/oai/")

model_id = "Alibaba-NLP/gte-Qwen1.5-7B-instruct/text"
input=[
    "apple",
    "orange",
    "pear",
    "watermelon",
    "grape",
    "strawberry",
    "banana",
    "lemon",
    "blueberry",
    "raspberry",
    "blackberry",
    "kiwi",
    "mango",
    "pineapple",
    "peach",
    "plum",
    "apricot",
    "cherry",
    "pomegranate",
    "fig",
    "date",
    "coconut",
]
fruit_embeddings_objs = client.embeddings.create(
    model=model_id,
    input=input
)
fruit_embeddings_raw = np.array([d.embedding for d in fruit_embeddings_objs.data])
fruit_embeddings_norms = np.linalg.norm(fruit_embeddings_raw, axis=1, keepdims=True)
fruit_embeddings = fruit_embeddings_raw / fruit_embeddings_norms

sample = "sour yellow"
sample_embedding_objs = client.embeddings.create(
    model=model_id,
    input=sample
)
sample_embedding = np.array([d.embedding for d in sample_embedding_objs.data])


k=5
sim_vec = np.dot(sample_embedding, fruit_embeddings.T)
most_similar_idxs = np.argsort(sim_vec, axis=1)[:, ::-1][:, :k].flatten().tolist()
print(f"Top {k} similar fruits to '{sample}':")
for i, idx in enumerate(most_similar_idxs):
    print(f"{i}. {input[idx]}: {sim_vec[:,idx]}")
```

Audio embeddings

```python
import io
import base64
import openai
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from sklearn import decomposition
import matplotlib.pyplot as plt

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def make_noisy_wav_dataurl() -> str:
    duration = 5
    sample_rate = 44100  # Standard sample rate
    num_samples = duration * sample_rate
    white_noise = np.random.uniform(-1, 1, num_samples)
    white_noise = (white_noise * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, white_noise)
    buffer.seek(0)
    wav_data = buffer.read()
    base64_encoded = base64.b64encode(wav_data).decode('utf-8')
    data_url = f"data:audio/wav;base64,{base64_encoded}"
    return data_url

def wav_file_to_dataurl(file_path: str) -> str:
    assert file_path.endswith(".wav"), "File must be a .wav file"
    with open(file_path, "rb") as f:
        wav_data = f.read()
    base64_encoded = base64.b64encode(wav_data).decode('utf-8')
    data_url = f"data:audio/wav;base64,{base64_encoded}"
    return data_url

spoken_wavs = [wav_file_to_dataurl(str(p)) for p in Path("data/spoken_wavs/").glob("*.wav")]
guitar_wavs = [wav_file_to_dataurl(str(p)) for p in Path("data/instrumental_wavs/").glob("*.wav")]
white_noise = [make_noisy_wav_dataurl() for _ in range(64)]
all_wavs = [*spoken_wavs, *guitar_wavs, *white_noise]

client = openai.Client(api_key="n/a", base_url="http://localhost:8080/oai/")

model_id = "openai/whisper-large-v2/audio"
audio_embeds_objs = sum(
    (
        client.embeddings.create(
            model=model_id,
            input=chunk
        ).data
        for chunk in chunker(all_wavs, 16)
    ),
    [])
audio_embeds_raw = np.array([d.embedding for d in audio_embeds_objs])
audio_embeds_raw_norms = np.linalg.norm(audio_embeds_raw, axis=1, keepdims=True)
audio_embeds = audio_embeds_raw / audio_embeds_raw_norms


## Plot pca for white noise vs spoken audio
pca = decomposition.PCA(n_components=2)
pca.fit(audio_embeds)
X = pca.transform(audio_embeds)
green = '#2ecc71'
orange = '#f39c12'
blue = '#3498db'
plt.figure(figsize=(8, 6))
plt.style.use('default')
plt.scatter(X[:len(spoken_wavs),0], X[:len(spoken_wavs),1], color=green, s=50, alpha=0.8, label='Spoken Audio (VoxCeleb2)')
plt.scatter(X[len(spoken_wavs):-len(white_noise),0], X[len(spoken_wavs):-len(white_noise),1], color=orange, s=50, alpha=0.8, label='Guitar Audio (MusicBench)')
plt.scatter(X[len(spoken_wavs)+len(guitar_wavs):,0], X[-len(white_noise):,1], color=blue, s=50, alpha=0.8, label='White Noise')
plt.title('PCA of whisper-large-v2 audio embeddings', fontsize=12, fontweight='bold', pad=20)
plt.xlabel('PC 1', fontsize=10, labelpad=10)
plt.ylabel('PC 2', fontsize=10, labelpad=10)
plt.legend(fontsize=12, loc='lower right', title='Audio Type')
plt.tight_layout()
plt.savefig('data/audio_embeds.png')
```
