#%%
import os
# from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.trainer import Trainer
import torch
import torch.nn.functional as F
# from torchvision.datasets import MNIST
# from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
import pickle
from functools import partial
import itertools
import plotly.graph_objects as go
from sklearn import metrics
#%%

def validationReport(net, predictions, targets, ignore):

    predictions = [net.int2char[id] for id in predictions.numpy()]
    targets = [net.int2char[id] for id in targets.numpy()]

    labels = [v for k,v in net.int2char.items() if k != ignore]


    plotConfusionMatrix(predictions, targets,labels)
    # report = metrics.classification_report(targets,predictions,labels=labels)

def plotConfusionMatrix(predictions, targets, labels):
    cm = metrics.confusion_matrix(targets, predictions, normalize="true", labels=labels)
    
    cm = cm * 100 # for percentage
    
    data = [[None for i in range(cm.shape[0])] for j in range(cm.shape[0])]

    if len(data) == 0:
        print(" --------------> Heatmap:no data exist in heatmap")

    # print(">> Dimension", len(data[0]), len(data))

    more_than = []
    all_alarms = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            if cm[i,j] > 0.01:
                data[i][j] = cm[i,j]

            if cm[i,j] >= 65 and i ==j:
                more_than.append((i,j, cm[i,j]))
            
            if i ==j:
                all_alarms.append((i,j, cm[i,j]))
    
    
    # print(f">> Alarms Accuracy: {[a[2] for a in sorted(all_alarms, key=lambda arg: arg[2])] }")
    
    snames_sub = [f"S{i}" for i,j,val in all_alarms]
    trace = go.Bar(x=snames_sub,y=[val for _,_, val in all_alarms])
    
    fig = go.Figure(data=trace) 
    # # updating the figure properties
    # fig.update_xaxes(title_text="Source Name")
    # fig.update_yaxes(title_text="Accuracy (%)", range=[0,100])
    fig.update_layout(
    # xaxis_tickfont_size=14,
    yaxis=dict(
        title='Accuracy (%)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Source Name",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    # barmode='group',
    height=600, 
    width=1200,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    # fig.show()

    print(f">> Total Source= {len(labels)}, Sources whose accuracies are more than 65% = {len(more_than)}")


    # HeatMap

    # temp_dat = [[1.0, 0.0, 0.0  ],[0.,   0.62, 0.38],[0.,   0.,   1.  ]]

    fig = go.Figure(data=go.Heatmap(
        z= data,
        colorscale='Greys',
        x = [f"S{i}" for i in range (len(labels))],
        y = [f"S{i}" for i in range (len(labels))],
        hoverongaps=False, 
        hovertemplate=None
    ))
    fig.update_layout(
        width=1200,
        height=1200,
        # xaxis_nticks=cm.shape[0],
        # yaxis_nticks=cm.shape[0],

        yaxis=dict(
        title='Predicted Label',
        titlefont_size=16,
        tickfont_size=14,
        ),
        xaxis=dict(
        title = "True Label",
        titlefont_size=16,
        tickfont_size=14,
        )
        
    )
    # fig.show()




#%%
class AlarmDataset(Dataset):
    def __init__(self,sequences,vocab2int):
        self.nsamples = len(sequences)
        self.vocab2int = vocab2int 
        encoded_sequences = [self._encode2Int(l) for l in sequences]
        self.X = [torch.tensor(l[:-1], dtype=torch.int32) for l in encoded_sequences]
        self.Y = [torch.tensor(l[1:], dtype=torch.int32) for l in encoded_sequences]

    def __getitem__(self, index: int):
        x = self.X[index]  
        y = self.Y[index]
        return x,y
    
    def __len__(self) -> int:
        return self.nsamples

    def _encode2Int(self,l): # refactor this function move it to init 
        return [self.vocab2int[e] for e in l]

class MyDataModule(pl.LightningDataModule):
    
    def __init__(self, data_path:str, batch_size:int):
        super().__init__()
        self.batch_size = batch_size
                
        dataset_dict = None
        with open(data_path, 'rb') as f: 
            dataset_dict = pickle.load(f)
        
        self.vocab = set(list(itertools.chain.from_iterable(dataset_dict['train']+dataset_dict['valid']))) 
        
        self.int2vocab = dict(enumerate(self.vocab))
        self.vocab2int = {v:k for k,v in self.int2vocab.items()}
        self.seq_length = dataset_dict['sequence-length']-1  

        self.train_dataset = AlarmDataset(dataset_dict['train'], self.vocab2int)
        self.valid_dataset = AlarmDataset(dataset_dict['valid'], self.vocab2int)
        # inputs,targets = self.valid_dataset[2]
        # print(inputs) 
        # print(targets)

    def setup(self, stage: None):
        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4,drop_last=True, pin_memory=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4,drop_last=True, pin_memory=True)


#%%
class AlarmRNN(LightningModule):
    
    def __init__(self,dm,embedding_dim,n_hidden=256, n_layers=2,drop_prob=0.5, lr=0.001):
        # super().__init__()
        super(AlarmRNN,self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = dm.batch_size 
        
        # creating character dictionaries
        self.seq_length = dm.seq_length
        self.chars = dm.vocab # vocab
        self.int2char = dm.int2vocab #dict(enumerate(self.chars))
        self.char2int = dm.vocab2int  #{ch: ii for ii, ch in self.int2char.items()}
                
        ## TODO: define the layers of the model
        self.h = None
        self.embedding = torch.nn.Embedding(len(self.chars), embedding_dim)
        self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,dropout=self.drop_prob, batch_first=True)
        self.droput = torch.nn.Dropout(p=self.drop_prob)
        self.fc3 = torch.nn.Linear(in_features=self.n_hidden, out_features=len(self.chars))
        self.softmax = torch.nn.LogSoftmax(dim=1)  
    
    def __init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        device = None 
        if (torch.cuda.is_available()):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu") 
        
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.n_hidden).zero_().to(device)
        return hidden

    def initialize_hidden(self):
        self.h = self.__init_hidden()
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        x = x.long()
        embeds = self.embedding(x)
        out, hidden = self.gru(embeds,hidden)
        # Contiguous variables: If you are stacking up multiple LSTM outputs, it may be necessary to use .contiguous() to reshape the output.
        out = out.contiguous().view(-1,self.n_hidden) 
        out = self.fc3(out)
        out = self.softmax(out)
        # return the final output and the hidden state
        return out, hidden
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.000001, weight_decay=0.0000001)
        return optimizer
    
    # def on_epoch_start(self):
    #     self.initialize_hidden()
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        self.h = self.h.data # for GRU
        y_hat, self.h = self(x,self.h)
        
        loss = F.nll_loss(y_hat, y.view(self.batch_size*self.seq_length).long(), ignore_index=self.char2int["NoName"])    
        result = pl.TrainResult(loss) # logging
        return result
    
    def validation_step(self,batch, batch_idx):
        x,y = batch
        self.h = self.h.data
        y_hat, self.h = self(x,self.h)
        loss = F.nll_loss(y_hat, y.view(self.batch_size*self.seq_length).long(), ignore_index=self.char2int["NoName"])
        
        _, y_hat = torch.max(y_hat,dim=1) # probs are the indexes
        y_hat = y_hat.to("cpu")
        
        y = y.view(self.batch_size*self.seq_length).long().to('cpu')
        
        # result = pl.EvalResult()
        # result.log('val_loss', loss)
        # result.log('y_hat', y_hat)
        # result.log('y',y)


        return {'val_loss':loss, 'y':y, 'y_hat':y_hat}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([d['val_loss'] for d in outputs]).mean()
        y_hats = torch.cat([d["y_hat"] for d in outputs]) 
        ys = torch.cat([d['y'] for d in outputs]) # targets: real label

        validationReport(net=self, predictions=y_hats, targets=ys,ignore=self.char2int["NoName"] )
        print(f"> Average Valid Loss= {avg_loss}")

        result = pl.EvalResult()
        result.log('avg_val_loss', avg_loss)
        return result




#%%
n_hidden=1024
n_layers=5
n_epochs = 200 # start small if you are just testing initial behavior
batch_size = 64
embedding_dim = 512
drop_prob = 0.1


data_path = "../.data/raw-dataset-15-mins_prof.dataset"
dm = MyDataModule(data_path=data_path, batch_size=batch_size) 
#%%
model = AlarmRNN(dm,embedding_dim=embedding_dim,n_hidden=n_hidden,n_layers=n_layers,drop_prob=drop_prob)
model.initialize_hidden()

trainer = Trainer(amp_level='O2', precision=16,gpus=1,gradient_clip_val=0.5,max_epochs=800,check_val_every_n_epoch=10)
trainer.fit(model,dm.train_dataloader(), dm.val_dataloader())


# %%
