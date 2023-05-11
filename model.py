from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch_geometric.nn import GCNConv, GATConv, GraphConv, RGATConv, RGCNConv, FastRGCNConv, GATv2Conv, FiLMConv
import pdb


class BaseGraphModel(pl.LightningModule):
    def __init__(self, general_cfg, model_cfg, n_classes):
        # torch.manual_seed(1234)
        super().__init__()

        self.general_cfg = general_cfg
        self.mode_config = model_cfg
        
        self.init_common_layers(general_cfg, model_cfg, n_classes)
        self.init_lightning_stuff(general_cfg, n_classes)


    def init_common_layers(self, general_cfg, model_cfg, n_classes):
        self.x_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.y_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.w_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.h_embedding = nn.Embedding(num_embeddings=general_cfg['model']['emb_range']+1, embedding_dim=general_cfg['model']['emb_dim'])
        self.linear_prj = nn.Linear(in_features=general_cfg['model']['text_feature_dim'], out_features=general_cfg['model']['emb_dim']*6)


    def init_lightning_stuff(self, general_cfg, n_classes):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=general_cfg['training']['label_smoothing'])
        self.train_f1 = torchmetrics.F1Score(task='multiclass', threshold=0.5, num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', threshold=0.5, num_classes=n_classes)


    def configure_optimizers(self):
        base_lr = self.general_cfg['training']['base_lr']
        opt = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=self.general_cfg['training']['weight_decay'])

        if self.general_cfg['training']['use_warmup']:
            num_warmpup_epoch = self.general_cfg['training']['warmup_ratio'] * self.general_cfg['training']['num_epoch']
            def lr_foo(epoch):
                if epoch <= num_warmpup_epoch:
                    return 0.75 ** (num_warmpup_epoch - epoch)
                else:
                    return 0.97 ** (epoch - num_warmpup_epoch)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lr_foo
            )
            return [opt], [scheduler]
        
        return opt
    

    def calc_gnn_input(self, x_indexes, y_indexes, text_features):
        # calc spatial position embedding
        left_emb = self.x_embedding(x_indexes[:, 0])    # (n_nodes, embed_size)
        right_emb = self.x_embedding(x_indexes[:, 1])
        w_emb = self.w_embedding(x_indexes[:, 2])
        top_emb = self.y_embedding(y_indexes[:, 0])
        bot_emb = self.y_embedding(y_indexes[:, 1])
        h_emb = self.h_embedding(y_indexes[:, 2])
        pos_emb = torch.concat([left_emb, right_emb, w_emb, top_emb, bot_emb, h_emb], dim=-1)

        return pos_emb + self.linear_prj(text_features)
    

    def common_step(self, batch, batch_idx):
        x_indexes, y_indexes, text_features, edge_index, edge_type, labels = batch
        # print('x_index: ', x_indexes[:, 0].dtype, x_indexes[:, 0].shape)
        # print('y_index: ', y_indexes.dtype, y_indexes.shape)
        # print('text_features: ', text_features.dtype, text_features.shape)
        # print('edge_indexes: ', edge_index.dtype, edge_index.shape)
        # print('edfe_type: ', edge_type.dtype, edge_type.shape)
        # print('labels: ', labels.dtype, labels.shape)
        logits = self.forward(x_indexes, y_indexes, text_features, edge_index, edge_type)
        loss = self.criterion(logits, labels)
        return logits, loss, labels
    

    def training_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch, batch_idx)
        # print('logits shape: ', logits.shape)
        # print('labels shape: ', labels.shape)
        # pdb.set_trace()
        self.train_f1(torch.argmax(logits, dim=-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        logits, loss, labels = self.common_step(batch, batch_idx)
        self.val_f1(torch.argmax(logits, dim=-1), labels)
        self.log_dict({
            'val_loss': loss,
            'val_f1': self.val_f1
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_start(self) -> None:
        print('\n')



class RGCN_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        # torch.manual_seed(1234)
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    

    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            RGCNConv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                num_relations=4
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                RGCNConv(
                    in_channels=model_cfg['channels'][i], 
                    out_channels=model_cfg['channels'][i+1], 
                    num_relations=4
                )
            )
        self.classifier = nn.Linear(in_features=model_cfg['channels'][-1], out_features=n_classes)
        
        
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer in self.gnn_layers:
            x = layer(x, edge_index.to(torch.int64), edge_type)
            x = F.relu(x)
            # x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        logits = self.classifier(x)
        return logits
    

class GNN_FiLM_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        # torch.manual_seed(1234)
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    

    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            FiLMConv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                num_relations=4,
                act=None
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                FiLMConv(
                    in_channels=model_cfg['channels'][i], 
                    out_channels=model_cfg['channels'][i+1], 
                    num_relations=4,
                    act=None
                )
            )
        self.classifier = nn.Linear(in_features=model_cfg['channels'][-1], out_features=n_classes)
        
        
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer in self.gnn_layers:
            x = layer(x, edge_index.to(torch.int64), edge_type)
            x = F.relu(x)
            # x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        logits = self.classifier(x)
        return logits
    


class GATv2_Model(BaseGraphModel):
    def __init__(self, general_cfg, model_cfg, n_classes):
        # torch.manual_seed(1234)
        super().__init__(general_cfg, model_cfg, n_classes)
        self.init_gnn_layers(general_cfg, model_cfg, n_classes)
    

    def init_gnn_layers(self, general_cfg, model_cfg, n_classes):
        self.gnn_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=general_cfg['model']['emb_dim']*6, 
                out_channels=model_cfg['channels'][0], 
                heads=model_cfg['num_heads'],
                dropout=model_cfg['attn_dropout'],
                concat=model_cfg['concat']
            )
        ])
        for i in range(len(model_cfg['channels'])-1):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=model_cfg['channels'][i] if not model_cfg['concat'] else model_cfg['channels'][i]*model_cfg['num_heads'], 
                    out_channels=model_cfg['channels'][i+1], 
                    heads=model_cfg['num_heads'],
                    dropout=model_cfg['attn_dropout'],
                    concat=model_cfg['concat']
                )
            )
        self.classifier = nn.Linear(
            in_features=model_cfg['channels'][-1] if not model_cfg['concat'] else model_cfg['channels'][-1]*model_cfg['num_heads'], 
            out_features=n_classes
        )
        
    
    def forward(self, x_indexes, y_indexes, text_features, edge_index, edge_type):
        x = self.calc_gnn_input(x_indexes, y_indexes, text_features)
        for layer in self.gnn_layers:
            x = layer(x, edge_index.to(torch.int64))
            x = F.relu(x)
            # x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        x = F.dropout(x, p=self.general_cfg['model']['dropout_rate'])
        logits = self.classifier(x)
        return logits



if __name__ == '__main__':
    import yaml
    import os

    # load config and setup
    with open("configs/train_cfg.yaml") as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    model_type = general_cfg['options']['model_type']
    with open(os.path.join('configs', model_type+'.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = RGCN_Model(general_cfg, model_cfg, n_classes=2)
    pdb.set_trace()