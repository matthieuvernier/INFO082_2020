#!/usr/bin/env python
# coding: utf-8

# # Tutorial 4 - CNN para la clasificación de textos
# 
# ## 1. Dataset
# 
# Tenemos a disposición una gran cantidad de comentarios de Wikipedia que han sido etiquetados por evaluadores humanos por su comportamiento tóxico. Los tipos de toxicidad son:
# 
# - toxic (tóxico)
# - severe_toxic (muy tóxico)
# - obscene (obsceno)
# - threat (amenasa)
# - insult (insulto)
# - identity_hate (odio)

# In[ ]:


import pandas as pd


# ## 2. Clasificar textos según su toxicidad con una red convolucional
# 
# **Tarea**: Queremos aprender un modelo capaz de distinguir los textos tóxicos y no tóxico. Se trata de una clasificación binaria (columna "toxicity").

# ### 2.1 Leer el dataset

# In[ ]:


import torch
import spacy
import random
import torchtext
from torchtext import data
from torchtext import datasets


# In[ ]:


#!pip install torch==1.5.1


# In[ ]:


print(torch.__version__,spacy.__version__,torchtext.__version__)


# In[ ]:


TEXT = data.Field(tokenize='spacy', batch_first = True)
TOXIC = data.LabelField(dtype = torch.float)


# In[ ]:


fields = [(None, None),(None, None),('comment_text', TEXT),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),(None, None),('toxicity', TOXIC)]


# Se leen los CSV para tokenizarlos con Torchtext.data

# In[ ]:


import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '.',
                                        train = 'train_data_small.csv',
                                        validation= 'valid_data_small.csv',
                                        test = 'test_data_small.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
)

# In[ ]:


BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort_key=lambda x:len(x.toxicity),
    sort_within_batch=False)


# ### 2.2 Crear la arquitectura CNN
# 
# Empezamos por cargar vectores de palabras para el inglés.
# 
# (Para cargar sus propios vectores, por ejemplo para procesor otros idiomas, se puede inspirarse de: https://www.innoq.com/en/blog/handling-german-text-with-torchtext/)

# In[ ]:


MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

TOXIC.build_vocab(train_data)


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        
        #embedded = [batch size, emb dim, sent len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


# In[ ]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# ### 2.3 Funciones para optimizar el modelo

# In[ ]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# In[ ]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.comment_text).squeeze(1)
        
        loss = criterion(predictions, batch.toxicity)
        
        acc = binary_accuracy(predictions, batch.toxicity)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ### 2.4 Funciones para evaluar el modelo

# In[ ]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[ ]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
           
            predictions = model(batch.comment_text).squeeze(1)
            
            loss = criterion(predictions, batch.toxicity)
            
            acc = binary_accuracy(predictions, batch.toxicity)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# ### 2.5 Optimización del modelo

# In[ ]:


N_EPOCHS = 2

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        nombre = './toxic-model-CNN'+'_ep'+str(epoch+1)+'.pt'
        torch.save({'epoca': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'Valid_loss': best_valid_loss}, nombre)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# ### 2.6 Evaluación del modelo

# In[ ]:


best_model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

best_model.embedding.weight.data.copy_(pretrained_embeddings)
best_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
best_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[ ]:


name = './toxic-model-CNN'+'_ep'+str(2)+'.pt'
best_model.load_state_dict(torch.load(name)['model_state_dict'])


# In[ ]:


from sklearn.metrics import f1_score,confusion_matrix, classification_report


# In[ ]:


prediction_test = []
labels_test=[]
for batch in test_iterator:
    labels_test.append(batch.toxicity.cpu().detach().numpy())
    predictions = best_model(batch.comment_text.cpu()).squeeze(1)
    rounded_preds = torch.round(torch.sigmoid(predictions))
    prediction_test.append(rounded_preds.detach().numpy())
    

y_true = np.concatenate(labels_test)
y_pred = np.concatenate(prediction_test)


# In[ ]:


display(y_pred,y_true)


# In[ ]:


cm = confusion_matrix(y_true, y_pred)
display(cm)

print(classification_report(y_true, y_pred))
