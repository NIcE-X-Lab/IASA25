import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

def get_model(model_type, model_params, input_dim, output_dim=1):
    if model_type == "vgg16":
        return VGG16(num_classes=output_dim, **model_params)
    elif model_type == "cnnlstm":
        return CNNLSTM(input_dim=input_dim, output_dim=output_dim, **model_params)
    elif model_type == "alexnet":
        return AlexNet(num_classes=output_dim, **model_params)
    elif model_type == "GRU":
        return GRU(input_dim=input_dim, output_dim=output_dim, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class VGG16(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, dropout=0.3):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 750)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2).permute(0, 2, 1)
        pred = self.regressor(x).squeeze(-1)
        return pred

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        u = torch.tanh(x)
        scores = torch.matmul(u, self.attention_weights)
        attention_weights = torch.softmax(scores, dim=1)
        attended = x * attention_weights.unsqueeze(-1)
        return attended

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, attention_dim=None, output_dim=1, bidirectional=True, num_layers=1, use_attention=False, target_length=750):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.target_length = target_length
        if input_dim == 768:
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.temporal_align = None
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(0.3)
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.bn_gru = nn.BatchNorm1d(gru_output_dim)
        if attention_dim is not None and use_attention:
            self.proj = nn.Linear(gru_output_dim, attention_dim)
            self.bn_proj = nn.BatchNorm1d(attention_dim)
            self.regressor = nn.Linear(attention_dim, output_dim)
        else:
            self.proj = None
            self.bn_proj = None
            self.regressor = nn.Linear(gru_output_dim, output_dim)

    def forward(self, x, return_repr=True):
        if self.temporal_align is not None:
            x = x.transpose(1, 2)
            x = self.temporal_align(x)
            x = x.transpose(1, 2)
        out, _ = self.gru(x)
        B, T, H = out.shape
        out = self.bn_gru(out.contiguous().view(B * T, H)).view(B, T, H)
        out = self.dropout(out)
        if self.proj is not None:
            feat = self.proj(out)
            feat = self.bn_proj(feat.view(B * T, -1)).view(B, T, -1)
        else:
            feat = out
        if return_repr:
            return feat
        else:
            preds = self.regressor(feat).squeeze(-1)
            return preds

class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        embedding_dim=128,
        bidirectional=True,
        kernel_size_conv=3,
        padding_conv=1,
        stride_conv=1,
        bias_conv=True,
        dilation_conv=1,
        groups_conv=40,
        target_length=750,
        return_padded=False,
    ):
        super(CNNLSTM, self).__init__()
        self.bidirectional = bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_length = target_length
        self.input_norm = nn.InstanceNorm1d(num_features=input_dim, momentum=0.01, affine=True)
        if input_dim == 768:
            self.groups_conv = 1
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.groups_conv = groups_conv
            self.temporal_align = None
        self.cnn = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=kernel_size_conv,
            stride=stride_conv,
            padding=padding_conv,
            dilation=dilation_conv,
            groups=self.groups_conv,
            bias=bias_conv,
        )
        self.lstm_input_dim = input_dim * 2
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),
        )
        self.fc1 = nn.Linear(lstm_output_dim, embedding_dim)

    def forward(self, x, lengths=None, hidden=None, memory=None, return_padded=False):
        batch_size, seq_len, feat_dim = x.shape
        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        x = x.permute(0, 2, 1)
        x = self.input_norm(x)
        x_cnn = self.cnn(x)
        x_comb = torch.cat([x, x_cnn], dim=1)
        if self.temporal_align is not None:
            x_aligned = self.temporal_align(x_comb)
            new_lengths = torch.full((batch_size,), self.target_length, dtype=torch.long, device=x.device)
        else:
            x_aligned = x_comb
            new_lengths = lengths
        x_aligned = x_aligned.permute(0, 2, 1)
        packed = rnn_utils.pack_padded_sequence(x_aligned, new_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden_out, memory_out) = self.lstm(packed, (hidden, memory) if hidden is not None else None)
        padded, _ = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
        preds = self.regressor(padded).squeeze(-1)
        last_hidden = padded[:, -1, :]
        emb1 = self.fc1(last_hidden)
        if return_padded:
            return preds, emb1, hidden_out, memory_out, padded
        return preds, emb1, hidden_out, memory_out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim,
                 num_layers=2, use_attention=True, bidirectional=True, target_length=750):
        super(LSTM, self).__init__()
        self.use_attention = use_attention
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.target_length = target_length

        if input_dim == 768:
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.temporal_align = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.lstm_output_dim = lstm_output_dim
        self.bn_lstm = nn.BatchNorm1d(lstm_output_dim)

        if self.use_attention:
            self.attention = AttentionLayer(self.lstm_output_dim)

        self.fc1 = nn.Linear(self.lstm_output_dim, attention_dim)
        self.bn1 = nn.BatchNorm1d(attention_dim)
        self.fc2 = nn.Linear(attention_dim, output_dim)

    def forward(self, x, lengths):
        if self.temporal_align is not None:
            x = x.transpose(1, 2)
            x = self.temporal_align(x)
            x = x.transpose(1, 2)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        B, T, F = out.shape
        out = self.bn_lstm(out.contiguous().view(B * T, F)).view(B, T, F)
        if self.use_attention:
            out = self.attention(out)
        out = self.fc1(out)
        out = self.bn1(out.contiguous().view(-1, out.shape[-1])).view(B, T, -1)
        out = self.dropout(out)
        preds = self.fc2(out).squeeze(-1)
        return preds

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, bidirectional=True, target_length=750, 
                 kernel_size_conv=3, padding_conv=1, stride_conv=1, dilation_conv=1, bias_conv=True, groups_conv=1):
        super(CNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.target_length = target_length

        if input_dim == 768:  
            self.temporal_align = nn.AdaptiveAvgPool1d(target_length)
        else:
            self.temporal_align = None

        self.tcnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size_conv, padding=padding_conv, 
                     stride=stride_conv, dilation=dilation_conv, bias=bias_conv, groups=groups_conv),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size_conv, padding=padding_conv, 
                     stride=stride_conv, dilation=dilation_conv, bias=bias_conv, groups=groups_conv),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size_conv, padding=padding_conv, 
                     stride=stride_conv, dilation=dilation_conv, bias=bias_conv, groups=groups_conv),
            nn.ReLU(),
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, lengths):
        if self.temporal_align is not None:
            x = x.transpose(1, 2)
            x = self.temporal_align(x)
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.tcnn(x)
        x = x.transpose(1, 2)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        preds = self.fc(out).squeeze(-1)
        return preds


class AlexNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=11, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.temporal_pool = nn.AdaptiveAvgPool2d((750, 1))
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.temporal_pool(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        pred = self.regressor(x).squeeze(-1)
        return pred
    
    
class LearnableThreshold(nn.Module):
    def __init__(self, init_logit_threshold: float = 0.0):
        super().__init__()
        self.threshold_logit = nn.Parameter(torch.tensor(float(init_logit_threshold)))

    def shift_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits - self.threshold_logit
    
    
class LSTMBreakDetector(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, attention_dim: int = 64, num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True, target_length: int = 750):
        super().__init__()
        self.backbone = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            output_dim=1,
            num_layers=num_layers,
            use_attention=True,
            bidirectional=bidirectional,
            target_length=target_length,
        )
        self.learnable_threshold = LearnableThreshold(init_logit_threshold=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long)
        logits = self.backbone(x, lengths)
        shifted = self.learnable_threshold.shift_logits(logits)
        probs = torch.sigmoid(shifted)
        return probs


class TwoStageCNNLSTM(nn.Module):
    def __init__(self, hidden_dim_stage2: int = 128, target_length: int = 750,
                 num_layers_stage2: int = 2, bidirectional_stage2: bool = True,
                 kernel_size_conv: int = 3, padding_conv: int = 1, stride_conv: int = 1,
                 dilation_conv: int = 1, bias_conv: bool = True, groups_conv: int = 1,
                 fused_feature: str = "mfb", dropout_stage2: float = 0.3):
        super(TwoStageCNNLSTM, self).__init__()

        self.break_predictor = LSTMBreakDetector(
            input_dim=40, hidden_dim=128, attention_dim=64, num_layers=2, dropout=0.3, bidirectional=True,
            target_length=750
        )

        assert fused_feature in ("mfb", "mfcc")
        self.fused_feature = fused_feature
        self.dropout2 = nn.Dropout(p=dropout_stage2)

        self.cnnlstm = CNNLSTM(
            input_dim=808,
            hidden_dim=hidden_dim_stage2,
            num_layers=num_layers_stage2,
            output_dim=1,
            bidirectional=bidirectional_stage2,
            kernel_size_conv=kernel_size_conv,
            padding_conv=padding_conv,
            stride_conv=stride_conv,
            dilation_conv=dilation_conv,
            bias_conv=bias_conv,
            groups_conv=groups_conv,
            target_length=target_length,
        )

    def forward(self, mfb_features, wav2vec2_features, mfcc_features):
        fused_low_level = mfb_features if self.fused_feature == "mfb" else mfcc_features
        break_probs = self.break_predictor(fused_low_level)
        t_embed = wav2vec2_features.size(1)
        attn = F.interpolate(break_probs.unsqueeze(1), size=t_embed, mode='linear', align_corners=False).squeeze(1)
        wav2vec2_attended = wav2vec2_features * attn.unsqueeze(-1)
        low_level_attended = fused_low_level * attn.unsqueeze(-1)
        fused = torch.cat([wav2vec2_attended, low_level_attended], dim=-1)
        fused = self.dropout2(fused)
        lengths = torch.full((fused.size(0),), t_embed, dtype=torch.long)

        preds, _, _, _ = self.cnnlstm(fused, lengths)
        return break_probs, preds


class TwoStageGRU(nn.Module):
    def __init__(self, hidden_dim_stage2: int = 128, target_length: int = 750,
                 num_layers_stage2: int = 1, bidirectional_stage2: bool = True,
                 use_attention_stage2: bool = False, attention_dim_stage2: int | None = None,
                 fused_feature: str = "mfb", num_classes: int = 4, dropout_stage2: float = 0.3):
        super(TwoStageGRU, self).__init__()

        self.break_predictor = LSTMBreakDetector(
            input_dim=40, hidden_dim=128, attention_dim=64, num_layers=2, dropout=0.3, bidirectional=True,
            target_length=750
        )

        assert fused_feature in ("mfb", "mfcc")
        self.fused_feature = fused_feature
        self.dropout2 = nn.Dropout(p=dropout_stage2)

        self.gru = GRU(
            input_dim=808,
            hidden_dim=hidden_dim_stage2,
            attention_dim=attention_dim_stage2,
            output_dim=1,
            bidirectional=bidirectional_stage2,
            num_layers=num_layers_stage2,
            use_attention=use_attention_stage2,
            target_length=target_length
        )

    def forward(self, mfb_features, wav2vec2_features, mfcc_features):
        fused_low_level = mfb_features if self.fused_feature == "mfb" else mfcc_features
        break_probs = self.break_predictor(fused_low_level)
        t_embed = wav2vec2_features.size(1)
        attn = F.interpolate(break_probs.unsqueeze(1), size=t_embed, mode='linear', align_corners=False).squeeze(1)
        wav2vec2_attended = wav2vec2_features * attn.unsqueeze(-1)
        low_level_attended = fused_low_level * attn.unsqueeze(-1)
        fused = torch.cat([wav2vec2_attended, low_level_attended], dim=-1)
        fused = self.dropout2(fused)
        preds = self.gru(fused, return_repr=False)
        return break_probs, preds