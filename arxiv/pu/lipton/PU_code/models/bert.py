from transformers import DistilBertForSequenceClassification, DistilBertModel
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

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

class CodeBertClassifier(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x, **kwargs):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs.logits

def initialize_codebert_based_model(net, num_classes):
    assert net == "microsoft/codebert-base"
    model = CodeBertClassifier.from_pretrained(net, num_labels=num_classes)
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    # encoder = encoder.to("cuda")
    # encoder.eval()

    # # freeze everything if you want
    # for p in encoder.parameters():
    #     p.requires_grad = False

    # def embed_texts(texts, batch_size=32):
    #     all_vecs = []

    #     for batch in tqdm(DataLoader(texts, batch_size=batch_size)):
    #         inputs = tokenizer(
    #             batch,
    #             padding=True,              # pad only within batch
    #             truncation=True,
    #             max_length=256,
    #             return_tensors="pt"
    #         ).to("cuda")

    #         with torch.inference_mode():     # faster than no_grad
    #             with torch.autocast("cuda"): # fp16
    #                 out = encoder(**inputs)
    #                 cls = out.last_hidden_state[:, 0, :]  # (B, 768)

    #         all_vecs.append(cls.cpu())

    #     return torch.cat(all_vecs).numpy()
    
    # # unlabeled_text = pd.read_parquet("/home/ubuntu/data/Task_A/test.parquet")["code"].tolist()
    # # labeled_text = pd.read_parquet("/home/ubuntu/data/Task_A/test_sample.parquet")["code"].tolist()
    # # texts = unlabeled_text + labeled_text # TODO concat the unlabeled / labeled test set
    # # print("embedding")
    # # X_train_emb = embed_texts(texts)

    # emb = model.roberta.embeddings.word_embeddings.weight.detach().cpu().numpy() # might have to use this instead of X_train_emb
    # print(emb.shape)

    # pca = PCA(n_components=200)
    # reduced = pca.fit_transform(emb)
    # reconstructed = pca.inverse_transform(reduced)
    # with torch.no_grad():
    #     model.roberta.embeddings.word_embeddings.weight.copy_(
    #         torch.tensor(reconstructed)
    #     )

    return model