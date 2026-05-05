from huggingface_hub import configure_http_backend
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def backend_factory():
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # force timeout globally
    original_request = session.request
    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", 16)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout
    return session

configure_http_backend(backend_factory=backend_factory)

from transformers import DistilBertForSequenceClassification, DistilBertModel

import os

os.environ["HF_HUB_TIMEOUT"] = "16"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "16"

class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


def initialize_bert_based_model(net, num_classes):

	if net == 'distilbert-base-uncased':
		model = DistilBertClassifier.from_pretrained(
			net,
			num_labels=num_classes)
	else:
		raise ValueError(f'Model: {net} not recognized.')
	return model