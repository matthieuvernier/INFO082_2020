#!/usr/bin/env python
# coding: utf-8

# # Tutorial 4 - CNN para la clasificación de textos
# 
# ## 1. Prepración del Dataset para extraer la categoría de las noticias
# 
# Tenemos a nuestra disposición el dataset de **CNN Chile** (16.472 noticias).
# 
# El dataset toma la forma de archivo CSV con la estructura siguiente:
# - ID, country, media_outlet, url, title, body, date
# 
# En un inicio, a partir de la URL vamos a extraer la categoría de la noticia.

# In[1]:


import pandas as pd

DATASET_CSV="../datasets/CNNCHILE_RAW.csv"

df = pd.read_csv(DATASET_CSV,sep=',',error_bad_lines=False)
df = df.drop(['Unnamed: 0'], axis = 1) # Para suprimir la columna ID
df['date'] = pd.to_datetime(df['date']) # Para convertir la columna date en formato datetime

df


# In[2]:


import re

for index, row in df.iterrows():
    url=row['url']
    obj = re.findall('(\w+)://([\w\-\.]+)/([\w\-]+).([\w\-]+)', url) 
    
    category=obj[0][2]
    
    df.loc[index,'category'] = category


# - ¿Cuáles son las categorias?

# In[ ]:


#!pip install --user pandasql


# In[10]:


from pandasql import sqldf


# In[11]:


q="""SELECT category, count(*) FROM df GROUP BY category ORDER BY count(*) DESC;"""
result=sqldf(q)
result


# - Guardaremos las categorias que contienen más de 1000 noticias y las noticias que tienen más de 5 caracteres

# In[31]:


q="""SELECT * FROM df WHERE category IN ('pais','deportes','tendencias','tecnologias','cultura','economia','mundo') ORDER BY date;"""
df_CNN=sqldf(q)
df_CNN


# In[32]:


q="""SELECT * FROM df_CNN WHERE length(body)>5"""
df_CNN=sqldf(q)
df_CNN


# Guardaremos los datos en tres archivos CSV: CNN_train, CNN_valid, CNN_test

# In[ ]:





# In[61]:


import numpy as np

valid, test, train = np.split(df_CNN, [int(.15*len(df_CNN)), int(.3*len(df_CNN))])


# In[63]:


print(df_CNN.shape)
print(train.shape)
print(valid.shape)
print(test.shape)


# In[68]:


train.to_csv("CNN_train.csv", encoding="UTF-8",index=False)
valid.to_csv("CNN_valid.csv", encoding="UTF-8",index=False)
test.to_csv("CNN_test.csv", encoding="UTF-8",index=False)


# ## 2. Clasificar textos según su categoria temática con una red convolucional
# 
# **Tarea**: Queremos aprender un modelo capaz de distinguir las noticias según su categoria.

# ### 2.1 Leer el dataset

# In[21]:


#!pip install --user torch
#!pip install --user torchtext
#!pip install --user spacy


# In[22]:


import torch
import spacy
import random
import torchtext
from torchtext import data
from torchtext import datasets


# In[26]:


spacy_es = spacy.load('es_core_news_sm')


# In[27]:


def tokenize_es(sentence):
    return [tok.text for tok in spacy_es.tokenizer(sentence)]


# In[28]:


print(torch.__version__,spacy.__version__,torchtext.__version__)


# In[29]:


TEXT = data.Field(tokenize=tokenize_es, batch_first = True)
CATEGORY = data.LabelField(dtype = torch.float)


# In[30]:


fields = [(None, None),(None, None),(None, None),(None, None),('body', TEXT),(None, None),('category', CATEGORY)]


# Se leen los CSV para tokenizarlos con Torchtext.data

# In[69]:


import numpy as np

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '.',
                                        train = 'CNN_train.csv',
                                        validation= 'CNN_valid.csv',
                                        test = 'CNN_test.csv',
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
)


# In[ ]:


#for i in range(50):
#    print(vars(valid_data[i])['body'].__len__())


# In[ ]:


#vars(test_data[12])


# In[72]:


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

# In[74]:


MAX_VOCAB_SIZE = 50000

vec = torchtext.vocab.Vectors('glove-sbwc.i25.vec.gz', cache='.')
TEXT.build_vocab(train_data, vectors=vec, max_size = MAX_VOCAB_SIZE, unk_init = torch.Tensor.normal_)

#TEXT.build_vocab(train_data, 
#                max_size = MAX_VOCAB_SIZE, 
#                 vectors = "glove.6B.100d", 
#                 unk_init = torch.Tensor.normal_)

CATEGORY.build_vocab(train_data)


# In[75]:


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


# In[79]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[80]:


INPUT_DIM


# In[81]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# ### 2.3 Funciones para optimizar el modelo

# In[82]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# In[97]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.body).squeeze(1)
        
        loss = criterion(predictions, batch.category)
        
        acc = binary_accuracy(predictions, batch.category)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[98]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ### 2.4 Funciones para evaluar el modelo

# In[99]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# In[100]:


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
        nombre = './tematic-model-CNN'+'_ep'+str(epoch+1)+'.pt'
        torch.save({'epoca': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'Valid_loss': best_valid_loss}, nombre)
    
    print("Epoch:"+str(epoch+1:02)+" | Epoch Time: "+str(epoch_mins)+"m "+str(epoch_secs)+"s")
    print("\tTrain Loss: "+str(train_loss:.3f)+" | Train Acc: "+str(train_acc*100:.2f)+"%")
    print("\t Val. Loss: "+str(valid_loss:.3f)+" |  Val. Acc: "+str(valid_acc*100:.2f)+"%"')


# ### 2.6 Evaluación del modelo

# In[57]:


best_model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[58]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

best_model.embedding.weight.data.copy_(pretrained_embeddings)
best_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
best_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[59]:


name = './tematic-model-CNN'+'_ep'+str(2)+'.pt'
best_model.load_state_dict(torch.load(name, map_location=torch.device('cpu'))['model_state_dict'])


# In[60]:


from sklearn.metrics import f1_score,confusion_matrix, classification_report


# In[61]:


prediction_test = []
labels_test=[]
for batch in test_iterator:
    labels_test.append(batch.category.cpu().detach().numpy())
    predictions = best_model(batch.body.cpu()).squeeze(1)
    rounded_preds = torch.round(torch.sigmoid(predictions))
    prediction_test.append(rounded_preds.detach().numpy())
    

y_true = np.concatenate(labels_test)
y_pred = np.concatenate(prediction_test)


# In[62]:


display(y_pred,y_true)


# In[63]:


cm = confusion_matrix(y_true, y_pred)
display(cm)

print(classification_report(y_true, y_pred))


# In[ ]:




