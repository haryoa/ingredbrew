from pytorch_lightning.core.lightning import LightningModule

from recibrew.data_util import construct_torchtext_iterator
from recibrew.nn.gru_bahdanau import Encoder, Decoder
from recibrew.nn.transformers import FullTransformer
import torch
from torch.optim import AdamW
from torchtext.data import Field


class TransformersLightning(LightningModule):
    """
    Using Transformer. Research environment
    """

    def __init__(self, train_csv='../data/processed/train.csv', dev_csv='../data/processed/dev.csv',
                 test_csv='../data/processed/test.csv', num_embedding=128, dim_feedforward=512, num_encoder_layer=4,
                 num_decoder_layer=4, dropout=0.3, padding_idx=1, lr=0.001, nhead=2):
        super().__init__()
        self.lr = lr
        self.constructed_iterator_field = \
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

    def predict_inference_src(self, src: str, max_len: int = 140) -> str:
        """
        Used on inference to predict source
        :param src: String text as source
        :param max_len: max length of the generator
        :return: predicted text
        """
        src_field, tgt_field = \
            self.constructed_iterator_field['src_field'], self.constructed_iterator_field['tgt_field']
        end_token_id = src_field.vocab.stoi['</s>']
        src_input = [src_field.vocab[x] for x in '<s> {} </s>'.format(src).split()]

        src_tensor = torch.LongTensor(src_input).unsqueeze(1).cuda()
        memory, src_pad_mask = self.full_transformer.forward_encoder(src_tensor)

        tgt_input = [tgt_field.vocab[x] for x in '<s>'.split()]

        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_input).unsqueeze(1).cuda()

            output = self.full_transformer.forward_decoder(tgt_tensor, memory,
                                                            src_pad_mask)
            out_token = output.argmax(2)[-1].item()
            tgt_input.append(out_token)
            if out_token == end_token_id:
                break
        return ' '.join([tgt_field.vocab.itos[x] for x in tgt_input])


class GRUBahdanauLightning(LightningModule):
    """
    GRU + Bahdanau attention, Research environment
    """

    def __init__(self, train_csv='../data/processed/train.csv', dev_csv='../data/processed/dev.csv',
                 test_csv='../data/processed/test.csv', lr=1e-3, gru_params=None, padding_idx=1,
                 max_len=140):
        """

        :param train_csv:
        :param dev_csv:
        :param test_csv:
        :param lr:
        :param gru_params: dict that contains:
            embedding_dim : word embedding
            hidden_dim    : hidden unit on decoder and encoder
            enc_bidirectional : boolean whether the GRU encoder is bidirectional or not
            enc_gru_layers : int , how many stack GRU in encoder

        :param padding_idx:
        :param max_len:
        """
        super().__init__()
        if gru_params is None:
            raise Exception('gru_params need to be filled')
        self.lr = lr
        self.constructed_iterator_field = \
            construct_torchtext_iterator(train_csv, dev_csv, test_csv, device='cuda', fix_length=None)
        self.vocab_size = len(self.constructed_iterator_field['src_field'].vocab)  # src and tgt have same vocabulary
        gru_params.update({'vocab_size': self.vocab_size})
        self.encoder = Encoder(**gru_params)
        self.decoder = Decoder(**gru_params)
        self._extract_gru_params(gru_params)
        self.shared_embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
                                                   padding_idx=padding_idx)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.max_len = max_len
        self.save_hyperparameters()

    def _extract_gru_params(self, gru_params):
        self.embedding_dim = gru_params.get('embedding_dim', None)
        self.vocab_size = gru_params.get('vocab_size', None)

    def forward_encoder(self, src):
        src_embedded = self.shared_embedding(src)
        enc_out, hidden = self.encoder(src_embedded)
        return enc_out, hidden

    def predict_inference_src(self, src: str, max_len: int = 140) -> str:
        """
        Used on inference to predict source
        :param src: String text as source
        :param max_len: max length of the generator
        :return: predicted text
        """
        src_field, tgt_field = \
            self.constructed_iterator_field['src_field'], self.constructed_iterator_field['tgt_field']
        end_token_id = src_field.vocab.stoi['</s>']
        src_input = [src_field.vocab[x] for x in '<s> {} </s>'.format(src).split()]
        gbl = self.eval()
        src_tensor = torch.LongTensor(src_input).unsqueeze(1).cuda()
        enc_out, hidden = gbl.forward_encoder(src_tensor)
        tgt_input = [tgt_field.vocab[x] for x in '<s>'.split()]
        tgt_tensor = torch.LongTensor(tgt_input).unsqueeze(1).cuda()

        for i in range(max_len):
            tgt_embedded = gbl.shared_embedding(tgt_tensor)
            output, hidden, _ = gbl.Ldecoder.forward(tgt_embedded, hidden, enc_out)
            out_token = output.argmax(1)[-1].item()
            tgt_input.append(out_token)
            tgt_tensor = torch.LongTensor([out_token]).unsqueeze(1).cuda()

            if out_token == end_token_id:
                break
        return ' '.join([tgt_field.vocab.itos[x] for x in tgt_input])

    def forward_decoder_train(self, tgt, hidden, enc_out):
        tgt_targets = tgt[1:, :]
        tgt_inputs = tgt[:-1, :]
        loss = 0
        counter = 0

        # Use teacher forcing
        for i in range(tgt_targets.shape[0]):
            tgt_input = tgt_inputs[i:i + 1, :]
            tgt_gold = tgt_targets[i, :]
            tgt_embedded = self.shared_embedding(tgt_input)
            pred, hidden, _ = self.decoder.forward(tgt_embedded, hidden, enc_out)
            loss += self.criterion.forward(pred, tgt_gold)
            counter += 1
        loss = loss / counter
        return loss

    def forward(self, src, tgt, train=True):
        if not train:
            raise NotImplementedError()
        else:
            enc_out, hidden = self.forward_encoder(src)
            loss = self.forward_decoder_train(tgt, hidden, enc_out)
            return loss

    def training_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.tgt
        loss = self.forward(src, tgt, train=True)
        return {'loss': loss}

    def train_dataloader(self):
        """
        Load Trainer data loader here
        """
        return self.constructed_iterator_field['train_iter']

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def val_dataloader(self):
        return self.constructed_iterator_field['val_iter']

    def validation_step(self, batch, batch_idx):
        src, tgt = batch.src, batch.tgt
        loss = self.forward(src, tgt, train=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': {'avg_loss': avg_loss.cpu().numpy()}}
