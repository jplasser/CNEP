import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from sklearn import metrics
import numpy as np
#import mimic3models.metrics as m
import matplotlib.pyplot as plt
from tqdm import tqdm

class LSTM_CNN2(nn.Module):

    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1):

        # dim, batch_norm, dropout, rec_dropout, task,
        # target_repl = False, deep_supervision = False, num_classes = 1,
        # depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True
        # self.dense = dense

        # some more parameters
        # self.output_dim = dim
        # self.batch_norm = batch_norm
        self.dropout = 0.3
        self.rec_dropout = 0.3
        self.depth = lstm_layers
        self.drop_conv = 0.5
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >= 2:
            self.lstm1 = nn.LSTM(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.layers - 1,
                                 dropout=self.rec_dropout,
                                 bidirectional=self.bidirectional,
                                 batch_first=True)
            self.do0 = nn.Dropout(self.dropout)

        # this is not in the original model
        # self.act1 = nn.ReLU()
        if self.layers >= 2:
            self.lstm2 = nn.LSTM(input_size=self.hidden_dim * 2,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)
        else:
            self.lstm2 = nn.LSTM(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)

        self.do1 = nn.Dropout(self.dropout)
        # self.bn0 = nn.BatchNorm1d(48 * self.hidden_dim*2)

        # three Convolutional Neural Networks with different kernel sizes
        nfilters = [2, 3, 4]
        nb_filters = 100
        pooling_reps = []

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=2,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=3,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=4,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.do2 = nn.Dropout(self.drop_conv)
        self.final = nn.Linear(6800, self.num_classes)

    def forward(self, inputs, labels=None):
        out = inputs
        if self.layers >= 2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)
        out = self.do1(out)

        pooling_reps = []

        pool_vecs = self.cnn1(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn2(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn3(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        out = self.do2(representation)
        out = self.final(out)

        return out


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTMNew(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        self.flatten_parameters() 
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class LSTM_CNN4(nn.Module):
    
    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1, dropout=0.3, dropout_w=0.3, dropout_conv=0.5):

        #dim, batch_norm, dropout, rec_dropout, task,
        #target_repl = False, deep_supervision = False, num_classes = 1,
        #depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN4, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True

        # some more parameters
        self.dropout = dropout
        self.rec_dropout = dropout_w
        self.depth = lstm_layers
        self.drop_conv = dropout_conv
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >=2:
            self.lstm1 = LSTMNew(input_size=self.input_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.layers-1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=self.bidirectional,
                                batch_first=True)
            self.do0 = nn.Dropout(self.dropout)
            
        # this is not in the original model
        if self.layers >=2:
            self.lstm2 = LSTMNew(input_size=self.hidden_dim*2,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)
        else:
            self.lstm2 = LSTMNew(input_size=self.input_dim,
                                hidden_size=self.hidden_dim*2,
                                num_layers=1,
                                dropoutw=self.rec_dropout,
                                dropout=self.rec_dropout,
                                bidirectional=False,
                                batch_first=True)
        
        # three Convolutional Neural Networks with different kernel sizes
        nfilters=[2, 3, 4]
        nb_filters=100
        pooling_reps = []

        self.cnn1 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=2,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()
            )
        
        self.cnn2 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=3,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()
            )
        
        self.cnn3 = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_dim*2, out_channels=nb_filters, kernel_size=4,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten()
            )

        self.encoder = nn.Sequential(
                #nn.ReLU(),
                nn.MaxPool1d(kernel_size=664, stride=8, padding=0),
                nn.Flatten()
            )

        self.do2 = nn.Dropout(self.drop_conv)
        self.final = nn.Linear(6800, self.num_classes)

    def forward(self, inputs, labels=None):
        out = inputs
        if self.layers >=2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)
        
        pooling_reps = []
        
        pool_vecs = self.cnn1(out.permute((0,2,1)))
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn2(out.permute((0,2,1)))
        pooling_reps.append(pool_vecs)
        
        pool_vecs = self.cnn3(out.permute((0,2,1)))
        pooling_reps.append(pool_vecs)
            
        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        encoding = self.encoder(representation)
        out = self.do2(representation)
        out = self.final(out)

        return out, encoding
    
# training loop of the LSTM model

def train(dataloader, model, optimizer, criterion, device):
    """
    main training function that trains model for one epoch/iteration cycle
    Args:
        :param dataloader: torch dataloader
        :param model: model to train
        :param optimizer: torch optimizer, e.g., adam, sgd, etc.
        :param criterion: torch loss, e.g., BCEWithLogitsLoss()
        :param device: the target device, "cuda" oder "cpu"
    """
    
    total_loss = []
    # initialize empty lists to store predictions and targets
    final_predictions = []
    final_targets = []
    
    # set model to training mode
    model.train()
    
    # iterate over batches from dataloader
    #for inputs, targets in tqdm(dataloader, desc="Train epoch"):
    for inputs, targets in tqdm(dataloader):
        
        # set inputs and targets
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        
        # clear the gradients
        optimizer.zero_grad()
        
        # forward pass of inputs through the model
        predictions, _ = model(inputs)
        
        # calculate the loss
        loss = criterion(predictions, targets.view(-1,1))
        #loss_ = loss + model.regularizer() / len(dataloader.dataset)
        
        total_loss.append(loss.item())
        # move predicitions and targets to list
        pred = predictions.detach().cpu().numpy().tolist()
        targ = targets.detach().cpu().numpy().tolist()
        final_predictions.extend(pred)
        final_targets.extend(targ)
        
        # compute gradienta of loss w.r.t. to trainable parameters of the model
        loss.backward()
        
        # single optimizer step
        optimizer.step()
        
    return total_loss, final_predictions, final_targets
        
def evaluate(dataloader, model, device):
    """
    main eval function
    Args:
        :param dataloader: torch dataloader for test data set
        :param model: model to evaluate
        :param device: the target device, "cuda" oder "cpu"
    """
    
    # initialize empty lists to store predictions and targets
    final_predictions = []
    final_targets = []
    
    # set model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        for inputs, targets in dataloader:
            # set inputs and targets
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            
            # make predictions
            predictions, _ = model(inputs)
            
            # move predicitions and targets to list
            predictions = predictions.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
    # return final predicitions and targets
    return final_predictions, final_targets


# trainer function
def trainer(dataloader_train, dataloader_val, modelclass=LSTM_CNN4, number_epochs=10, hidden_dim=16, lstm_layers=2, lr=1e-3,
            dropout=0.5, dropout_w=0.5, dropout_conv=0.5, best_loss=10000, best_accuracy=0, best_roc_auc=0, early_stopping=0,
            verbatim=False, filename=None):

    print("Start of training procedure.")
    
    if early_stopping == 0:
        early_stopping = number_epochs + 1
    early_stopping_counter = 0
    modelsignature = f"{number_epochs}_{hidden_dim}_{lstm_layers}_{lr}_{dropout}-{dropout_w}-{dropout_conv}"
    # create device depending which one is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fetch model
    model = modelclass(hidden_dim=hidden_dim, lstm_layers=lstm_layers,
                      dropout=dropout, dropout_w=dropout_w, dropout_conv=dropout_conv)

    # send model to device
    print(f"Moving model on to device {device}")
    model.to(device)
    print(model)

    # load existing state dict
    if filename:
        print(f"Loading dict state from file {filename}.")
        model.load_state_dict(torch.load(filename))

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=4e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200, 280], gamma=0.1)

    # initialize loss function
    loss = nn.BCEWithLogitsLoss()

    if verbatim:
        print("Training Model")
    train_loss_values = []
    val_loss_values = []

    # define threshold
    threshold = 0.5
    logit_threshold = torch.tensor (threshold / (1 - threshold)).log()

    for epoch in range(number_epochs):

        # train for one epoch
        error, outputs, targets = train(dataloader_train, model, optimizer, loss, device)
        train_loss_values.append(error)

        o = torch.tensor(outputs)

        o = o > logit_threshold
        accuracy = metrics.accuracy_score(targets, o)
        l = np.asarray(error)
        if verbatim:
            print(f"Epoch Train: {epoch}, Accuracy Score = {accuracy:.4f}, Loss = {l.mean():.4f}")

        # validation of the model
        outputs, targets = evaluate(dataloader_val, model, device)
        outputs = torch.tensor(outputs)

        o = outputs > logit_threshold
        accuracy = metrics.accuracy_score(targets, o)
        l = nn.BCEWithLogitsLoss()(outputs, torch.tensor(targets).detach().view(-1,1))
        val_loss_values.append(l)

        fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
        roc_auc = metrics.auc(fpr, tpr)
        if verbatim:
            print(f"Epoch Val: {epoch}, Accuracy Score = {accuracy:.4f} ({best_accuracy:.4f}), ROCAUC = {roc_auc:.4f} ({best_roc_auc:.4f}), Loss = {l.mean():.4f} ({best_loss:.4f})")
            print("-"*20)

        #scheduler.step(roc_auc)
        exp_lr_scheduler.step()

        if l < best_loss:
            best_loss = l
            # save model
            if verbatim:
                print("Saving model for best Loss...")
            torch.save(model.state_dict(), "./model_loss.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            # save model
            if verbatim:
                print("Saving model for ROC AUC...")
            early_stopping_counter = 0
            torch.save(model.state_dict(), "./model_roc_auc.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")
        else:
            early_stopping_counter += 1

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # save model
            if verbatim:
                print("Saving model...")
            torch.save(model.state_dict(), "./model_best.pth")
            torch.save(model.state_dict(), f"./model__{modelsignature}__epoch-{epoch}_loss-{l}_acc-{accuracy}_auc-{roc_auc}.pth")
        
        if early_stopping_counter > early_stopping:
            if verbatim:
                print("Early stopping done.")
            break
    
    return (best_loss, best_accuracy, best_roc_auc), train_loss_values, val_loss_values, modelsignature


def calcMetrics(model, dataloader_test, filename, title):
    # define threshold
    threshold = 0.5
    logit_threshold = torch.tensor (threshold / (1 - threshold)).log()
    device = next(model.parameters()).device

    model.load_state_dict(torch.load(filename))
    model.eval()
    
    print()
    print(title)
    print("=" * len(title))

    # validation of the model
    outputs, targets = evaluate(dataloader_test, model, device)
    outputs = torch.tensor(outputs)

    o = outputs > logit_threshold
    accuracy = metrics.accuracy_score(targets, o)
    print(metrics.classification_report(targets, o))

    l = nn.BCEWithLogitsLoss()(outputs, torch.tensor(targets).detach().view(-1,1))

    print(f"Accuracy Score = {accuracy}, Loss = {l.mean()}")
    print("-"*20)
    m.print_metrics_binary(targets, outputs.reshape(-1,))

    fpr, tpr, thresholds = metrics.roc_curve(targets, outputs)
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC AUC = ", roc_auc)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal threshold is:", optimal_threshold)

    print("Final report:")
    #logit_threshold = torch.tensor(threshold / (1 - optimal_threshold)).log()
    o = outputs > optimal_threshold #logit_threshold
    accuracy = metrics.accuracy_score(targets, o)

    print(metrics.classification_report(targets, o))
    print(f"Accuracy Score = {accuracy}, Loss = {l.mean()}")
    print("-" * 20)
    m.print_metrics_binary(targets, o.reshape(-1, ))

    return roc_auc, targets, outputs

def plotLoss(train_loss, val_loss):
    def rollavg_direct(a,n): 
        assert n%2==1
        b = a*0.0
        for i in range(len(a)) :
            b[i]=a[max(i-n//2,0):min(i+n//2+1,len(a))].mean()
        return b

    plt.figure(figsize=(10,10))
    plt.title('Train/Val Loss')
    plt.plot([np.asarray(l).mean() for l in train_loss], label="Train loss")
    #plt.plot([np.asarray(l).mean() for l in val_loss], label="Val loss")
    plt.plot(rollavg_direct(np.asarray([np.asarray(l).mean() for l in val_loss]),21), label="Val loss (rolling avg)")

    plt.legend(loc = 'upper right')
    plt.xlabel('# Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('plot-losscurve.png')
    
def plotAUC(targets, outputs):
    fpr, tpr, threshold = metrics.roc_curve(targets, outputs)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('plot-AUC.png')