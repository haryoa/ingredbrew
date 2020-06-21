from pytorch_lightning.core.lightning import LightningModule

from recibrew.data_util import construct_torchtext_iterator
from recibrew.nn.transformers import FullTransformer
import torch
from torch.optim import AdamW


class TransformersLightning(LightningModule):
    """
    Research environment
    """

    def __init__(self, train_csv='../data/processed/train.csv', dev_csv='../data/processed/dev.csv',
                 test_csv='../data/processed/test.csv', num_embedding=128, dim_feedforward=512, num_encoder_layer=4,
                 num_decoder_layer=4, dropout=0.3, padding_idx=1, lr=0.001, nhead=2):
        super().__init__()
        self.lr=lr
        self.constructed_iterator_field =\
            construct_torchtext_iterator(train_csv, dev_csv, test_csv, device='cuda', fix_length=None)
        num_vocab = len(self.constructed_iterator_field['src_field'].vocab)
        self.transformer_params = dict(num_embedding=num_embedding, dim_feedforward=dim_feedforward,
                                       num_decoder_layer=num_decoder_layer,
                                       num_encoder_layer=num_encoder_layer, dropout=dropout, padding_idx=padding_idx,
                                       num_vocab=num_vocab, nhead=nhead)
        self.save_hyperparameters()
        self.full_transformer = FullTransformer(**self.transformer_params)

    def forward(self, src, tgt):
        return self.full_transformer.forward(src, tgt)

    def validation_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.tgt
        logits = self.forward(src, tgt[:-1])  # Remember, tgt is the input to the decoder
        output_dim = logits.shape[-1]
        loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.padding_idx)
        loss = loss_criterion(logits.view(-1, output_dim), tgt[1:].view(-1))  # tgt[1:] is the ground truth
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': {'avg_loss': avg_loss.cpu().numpy()}}

    def val_dataloader(self):
        return self.constructed_iterator_field['val_iter']

    def training_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.tgt
        logits = self.forward(src, tgt[:-1])  # Remember, tgt is the input to the decoder
        output_dim = logits.shape[-1]
        loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.padding_idx)
        loss = loss_criterion(logits.view(-1, output_dim), tgt[1:].view(-1))  # tgt[1:] is the ground truth
        return {'loss': loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        """
        Load Trainer data loader here
        """
        return self.constructed_iterator_field['train_iter']

    # def test_step(self, batch, batch_idx):
    #     pass
    #
    # def test_epoch_end(self, outputs):
    #     pass

    # def test_dataloader(self):
    #     return self.constructed_iterator_field['test_iter']