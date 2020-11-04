#!/usr/bin/env python
# coding: utf-8

# # Tutorial 4 - CNN para la clasificación de textos
# 
# ## 1. Preparación del Dataset para extraer la categoría de las noticias
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

# In[3]:


from pandasql import sqldf


# In[4]:


q="""SELECT category, count(*) FROM df GROUP BY category ORDER BY count(*) DESC;"""
result=sqldf(q)
result


# - Guardaremos las categorias que contienen más de 1000 noticias y las noticias que tienen más de 5 caracteres

# In[5]:


q="""SELECT * FROM df WHERE category IN ('pais','deportes','tendencias','tecnologias','cultura','economia','mundo') ORDER BY date;"""
df_CNN=sqldf(q)
df_CNN


# In[6]:


q="""SELECT * FROM df_CNN WHERE length(body)>5 ORDER BY date"""
df_CNN=sqldf(q)
df_CNN


# Guardaremos los datos en tres archivos CSV: CNN_train, CNN_valid, CNN_test

# In[7]:


import numpy as np

valid, test, train = np.split(df_CNN, [int(.15*len(df_CNN)), int(.3*len(df_CNN))])


# In[8]:


print(df_CNN.shape)
print(train.shape)
print(valid.shape)
print(test.shape)


# In[9]:


train.to_csv("CNN_train.csv", encoding="UTF-8",index=False)
valid.to_csv("CNN_valid.csv", encoding="UTF-8",index=False)
test.to_csv("CNN_test.csv", encoding="UTF-8",index=False)


# ## 2. Clasificar textos según su categoria temática con una red convolucional
# 
# **Tarea**: Queremos aprender un modelo capaz de distinguir las noticias según su categoria.

# ### 2.1 Leer el dataset

# In[ ]:


#!pip install --user torch
#!pip install --user torchtext
#!pip install --user spacy


# In[10]:


import torch
import spacy
import random
import torchtext
from torchtext import data
from torchtext import datasets


# In[11]:


spacy_es = spacy.load('es_core_news_sm')


# In[12]:


def tokenize_es(sentence):
    return [tok.text for tok in spacy_es.tokenizer(sentence)]


# In[13]:


print(torch.__version__,spacy.__version__,torchtext.__version__)


# In[15]:


TEXT = data.Field(tokenize=tokenize_es, batch_first = True)
CATEGORY = data.LabelField()  # MULTICLASS -se borró el argumento "(dtype = torch.float)"


# In[16]:


fields = [(None, None),(None, None),(None, None),(None, None),('body', TEXT),(None, None),('category', CATEGORY)]


# Se leen los CSV para tokenizarlos con Torchtext.data

# In[17]:


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


# In[18]:


vars(train_data[-1])


# In[19]:


BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort_key=lambda x:len(x.category),
    sort_within_batch=False)


# ### 2.2 Crear la arquitectura CNN
# 
# Empezamos por cargar vectores de palabras para el inglés.
# 
# (Para cargar sus propios vectores, por ejemplo para procesor otros idiomas, se puede inspirarse de: https://www.innoq.com/en/blog/handling-german-text-with-torchtext/)

# In[21]:


#!wget http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz


# In[22]:


MAX_VOCAB_SIZE = 50000

## TENER VECTORES EN ESPAÑOL
vec = torchtext.vocab.Vectors('glove-sbwc.i25.vec.gz', cache='.')
TEXT.build_vocab(train_data, vectors=vec, max_size = MAX_VOCAB_SIZE, unk_init = torch.Tensor.normal_)

# TENER VECTORES EN INGLES
# TEXT.build_vocab(train_data, 
#                max_size = MAX_VOCAB_SIZE, 
#                 vectors = "glove.6B.100d", 
#                 unk_init = torch.Tensor.normal_)

CATEGORY.build_vocab(train_data)


# In[23]:


print(CATEGORY.vocab.stoi)


# In[47]:


import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        #text = text.permute(1, 0)
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)


# In[48]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
OUTPUT_DIM = len(CATEGORY.vocab) ##### MULTICLASS ---> la dimensión del output no es 1 (clasificación binaria)
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[49]:


OUTPUT_DIM


# In[50]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[51]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# ### 2.3 Funciones para optimizar el modelo

# In[52]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss() #MULTICLASS ---> en lugar de .BCEWithLogitsLoss() (Binary Cross Entropy)

model = model.to(device)
criterion = criterion.to(device)


# In[58]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        #print("Tamaño texto de entrada:"+str(batch.body.shape))
        
        predictions = model(batch.body)
        
        #print("Tamaño predecciones de salida:"+str(predictions.shape)) 
        
        #print("Tamaño target:"+str(batch.category.shape)) 
        
        loss = criterion(predictions, batch.category)
        
        acc = categorical_accuracy(predictions, batch.category)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[59]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ### 2.4 Funciones para evaluar el modelo

# In[60]:


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


# In[61]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.body)
            
            loss = criterion(predictions, batch.category)
            
            acc = categorical_accuracy(predictions, batch.category)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 
# ### 2.5 Optimización del modelo

# In[57]:


print("inicio optimización")

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        name = './tematic-model-CNN'+'_ep'+str(epoch+1)+'.pt'
        torch.save({'epoca': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'Valid_loss': best_valid_loss}, name)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# ### 2.6 Evaluación del modelo

# In[ ]:


best_model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)


# In[ ]:


pretrained_embeddings = TEXT.vocab.vectors
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

best_model.embedding.weight.data.copy_(pretrained_embeddings)
best_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
best_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# In[ ]:


name = './tematic-model-CNN'+'_ep'+str(2)+'.pt'
best_model.load_state_dict(torch.load(name, map_location=torch.device('cpu'))['model_state_dict'])


# In[ ]:


test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# In[ ]:


from sklearn.metrics import f1_score,confusion_matrix, classification_report


# In[ ]:


prediction_test = []
labels_test=[]
for batch in test_iterator:
    labels_test.append(batch.category.cpu().detach().numpy())
    predictions = best_model(batch.body.cpu()).squeeze(1)
    rounded_preds = torch.round(torch.sigmoid(predictions))
    prediction_test.append(rounded_preds.detach().numpy())
    

y_true = np.concatenate(labels_test)
y_pred = np.concatenate(prediction_test)


# In[ ]:


#display(y_pred,y_true)


# In[ ]:


cm = confusion_matrix(y_true, y_pred)
print(cm)

print(classification_report(y_true, y_pred))


# ## 2.7 Hacer predicciones con el modelo

# In[ ]:


def predict_class(model, sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()


# In[ ]:


pred_class = predict_class(model, "Real Madrid ganó el partido 3 a 0.")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

