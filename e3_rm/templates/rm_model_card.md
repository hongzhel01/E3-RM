---
{{ card_data }}
---

# Model Card for {{ model_name }}

This model is a fine-tuned version of [{{ base_model }}](https://huggingface.co/{{ base_model }}){% if dataset_name %} on the [{{ dataset_name }}](https://huggingface.co/datasets/{{ dataset_name }}) dataset{% endif %}.
It has been trained using [E3-RM](https://github.com/hongzhel01/E3-RM).

## Quick start

```python
from transformers import pipeline

text = "The capital of France is Paris."
rewarder = pipeline(model="{{ hub_model_id }}", device_map="auto")
output = rewarder(text)[0]
print(output["score"])
```

## Training procedure

{% if wandb_url %}[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>]({{ wandb_url }}){% endif %}
{% if trackio_url %}[<img src="https://raw.githubusercontent.com/gradio-app/trackio/refs/heads/main/trackio/assets/badge.png" alt="Visualize in Trackio" title="Visualize in Trackio" width="150" height="24"/>]({{ trackio_url }}){% endif %}
{% if comet_url %}[<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/master/logo/comet_badge.png" alt="Visualize in Comet" width="150" height="24"/>]({{ comet_url }}){% endif %}

This model was trained with {{ trainer_name }}{% if paper_id %}, a method introduced in [{{ paper_title }}](https://huggingface.co/papers/{{ paper_id }}){% endif %}.

### Framework versions

- E3-RM: {{ e3_rm_version }}
- Transformers: {{ transformers_version }}
- Pytorch: {{ pytorch_version }}
- Datasets: {{ datasets_version }}
- Tokenizers: {{ tokenizers_version }}

## Citations

{% if trainer_citation %}Cite {{ trainer_name }} as:

```bibtex
{{ trainer_citation }}
```{% endif %}
