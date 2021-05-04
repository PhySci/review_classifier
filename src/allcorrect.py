import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW
import torch
from tqdm import tqdm
from collections import defaultdict


BATCH_SIZE = 256
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
EPOCHS = 10
RANDOM_SEED = 12345
n_samples = None # 10*BATCH_SIZE


def get_device():
    """

    :return:
    """
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    return device


def load_data() -> pd.DataFrame:
    """

    :return:
    """
    data_pth = '../data/reviews.xlsx'
    try:
        df = pd.read_excel(data_pth, index_col='id', engine='openpyxl')
    except Exception as err:
        print('Can not read the data. Error message is %s', repr(err))
    print(df.shape)
    mapper = {'RL': 0, 'Rl': 0, 'L-': 1, 'l-': 1, 'YL': 2, 'yl': 2, 'Yl': 2, 'L+': 3}
    df['label'] = df['mark'].map(mapper)
    return df


class ReviewDataset(Dataset):

    def __init__(self, reviews, target, tokenizer, max_len=128):
        self._review = reviews
        self._target = target
        self._max_len = max_len
        self._tokenizer = tokenizer
        self._tokens = None
        self._process_reviews()

    def _process_reviews(self):
        """

        """
        params = {'add_special_tokens': True, 'max_length': self._max_len,
                  'return_token_type_ids': False, 'pad_to_max_length': True,
                  'return_attention_mask': True, 'return_tensors': 'pt'}

        self._tokens = self._review.apply(self._tokenizer.encode_plus, **params)

    def __len__(self):
        return self._review.shape[0]

    def __getitem__(self, item):

        tokens = self._tokens.iloc[item]

        result = {'text': self._review.iloc[item],
                  'tokens': tokens['input_ids'].flatten(),
                  'mask': tokens['attention_mask'].flatten(),
                  'target': self._target.iloc[item]}
        return result


class ReviewClassifier(nn.Module):

    def __init__(self, n_classes):
        super(ReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_output.pooler_output)
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """


    :param model:
    :param data_loader:
    :param loss_fn:
    :param optimizer:
    :param device:
    :param scheduler:
    :param n_examples:
    :return:
    """
    model.train()
    losses = []

    epoch_predictions = list()
    epoch_targets = list()

    for d in tqdm(data_loader):
        input_ids = d["tokens"].to(device)
        attention_mask = d["mask"].to(device)
        targets = d["target"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        losses.append(loss.item())

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        _, predictions = torch.max(outputs, dim=1)
        epoch_predictions.extend(predictions.cpu().numpy().tolist())
        epoch_targets.extend(d['target'].numpy().tolist())

    epoch_loss = np.mean(losses)
    epoch_report = classification_report(epoch_targets, epoch_predictions, output_dict=True)

    return epoch_report, epoch_loss


def eval_model(model, data_loader, loss_fn, device):
    """

    :param model:
    :param data_loader:
    :param loss_fn:
    :param device:
    :param n_examples:
    :return:
    """
    model = model.eval()

    losses = []
    epoch_predictions = list()
    epoch_targets = list()

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["tokens"].to(device)
            attention_mask = d["mask"].to(device)
            targets = d["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            _, predictions = torch.max(outputs, dim=1)
            epoch_predictions.extend(predictions.cpu().numpy().tolist())
            epoch_targets.extend(d['target'].numpy().tolist())

    epoch_loss = np.mean(losses)
    epoch_report = classification_report(epoch_targets, epoch_predictions, output_dict=True)

    return epoch_report, epoch_loss


def get_data_loader(df, tokenizer, batch_size=16, num_workers=2):
    t = ReviewDataset(df['review'], df['label'], tokenizer=tokenizer)
    data_loader = DataLoader(t, batch_size=batch_size, num_workers=num_workers)
    return data_loader


def main():

    device=get_device()
    print(device)

    df = load_data()
    if n_samples is None:
        t = df
    else:
        t = df.iloc[n_samples]
    df_train, df_test = train_test_split(t, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print('Train set size is {:d}'.format(df_train.shape[0]))
    print('Validation set size is {:d}'.format(df_test.shape[0]))

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_data_loader = get_data_loader(df_train, tokenizer, batch_size=BATCH_SIZE, num_workers=4)
    val_data_loader = get_data_loader(df_val, tokenizer, batch_size=BATCH_SIZE, num_workers=4)

    model = ReviewClassifier(n_classes=4)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_steps = 500 # len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_report, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler)
        val_report, val_loss = eval_model(model, val_data_loader, loss_fn, device)

        print(f'Train loss {train_loss}, validation loss {val_loss}')
        print(f'Classification report')
        print(pd.DataFrame.from_dict(val_report).transpose())
        print()

        history['train_report'].append(train_report)
        history['val_report'].append(val_report)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        #if val_acc > best_accuracy:
        #    torch.save(model.state_dict(), 'best_model_state.bin')
        #    best_accuracy = val_acc


if __name__ == '__main__':
    main()