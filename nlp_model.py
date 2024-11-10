from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

def map_range(x, a, b, c, d):
    return c + (d - c) * (x - a) / (b - a)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_bert = BertForSequenceClassification.from_pretrained("artemgoncarov/mezhnar_cp_nlp_model").to(device)
tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")

def predict_personality_traits(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model_bert(**inputs)
        predictions = outputs.logits.squeeze().cpu().numpy()  # Переводим на CPU и в numpy

    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    maps = {
        "Openness": [-1.156436, 0.89340734],
        "Conscientiousness": [-0.6558855, 1.3051943],
        "Extraversion": [-0.7169265, 1.2328148],
        "Agreeableness": [-0.4751214, 1.0410889],
        "Neuroticism": [-0.90752506, 1.1106781]
    }

    res = dict(zip(traits, predictions))
    subm = {}
    for key in res:
        subm[key] = map_range(res[key], maps[key][0], maps[key][1], 0, 1)

    return subm