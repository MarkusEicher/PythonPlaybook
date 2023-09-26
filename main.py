from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(["We are very happy to show you the great Transformers library.",
                    "We hope you don't hate it."])

# for result in results:
    #  print(result)



# Using the tokenizer manually

tokens = tokenizer.tokenize("We are very happy to show you the great Transformers library.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to show you the great Transformers library.")

# print(f'    Tokens: {tokens}')
# print(f'Token Ids: {token_ids}')
# print(f'Input Ids: {input_ids}')



# Using pretrained or own data

X_train = ["The sun is shining the whole day and people are very happy.",
            "Unfortunately the desk was closed and many folks were really pissed about that."]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
# print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)