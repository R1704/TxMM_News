from src import device, tokenizer, trained_model, pd, plt
from src.vad_regressor.config import *


def tokenize_sentence(sentence, tokenizer, max_tokens):
    """
    tokenizes the input for the model
    :param sentence: str
    :param tokenizer: huggingface tokenizer
    :param max_tokens: int
    :return: input_ids, attention_mask
    """
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_tokens,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    return input_ids, attention_mask


def predict_vad(sentence):
    """
    Predicts VAD values from a sentence, [v, a, d] values as numpy array which is then put in a pd.Series
    :param sentence: str
    :return: pd.Series([v, a, d])
    """
    input_ids, attention_mask = tokenize_sentence(sentence, tokenizer, MAX_TOKENS)
    pred = trained_model(input_ids, attention_mask)
    pred = pred[1].detach().cpu().numpy()[0]
    pred = pd.Series(pred)
    return pred


def plot_length(df):
    grouped = df.groupby('publish_date')['length'].mean()
    plt.plot(grouped)
    plt.show()

