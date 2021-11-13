import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Attention(nn.Module):
    def __init__(self, global_size, feature_size, attn_type='general'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        if attn_type == 'general':
            self.transform = nn.Linear(feature_size, global_size, bias=False)
        elif attn_type == 'general2':
            self.transform = nn.Linear(feature_size, global_size, bias=True)
    
    def forward(self, x, y, mask=None):
        # x: (seq_len, batch_size, global_size)
        # y: (batch_size, feature_size)
        if mask is None:
            mask = torch.ones(x.size(1), x.size(0)).to(x.device)
        # mask: (batch, seq_len)
        
        if self.attn_type == 'general':
            _x = x.permute(1, 2, 0)
            # _x: (batch_size, global_size, seq_len)
            _y = self.transform(y).unsqueeze(1)
            # _y: (batch_size, 1, global_size)
            alpha = F.softmax(torch.matmul(_y, _x), dim=2)
            # alpha: (batch_size, 1, seq_len)
        elif self.attn_type == 'general2':
            _x = x.permute(1, 2, 0)
            # _x: (batch_size, global_size, seq_len)
            _y = self.transform(y).unsqueeze(1)
            # _y: (batch_size, 1, global_size)
            alpha = F.softmax(torch.matmul(_y, _x) * mask.unsqueeze(1), dim=2)
            # alpha: (batch_size, 1, seq_len)
            alpha_mask = alpha * mask.unsqueeze(1)
            # alpha_mask: (batch_size, 1, seq_len)
            alpha_sum = torch.sum(alpha_mask, dim=2, keepdim=True)
            # alpha_sum: (batch_size, 1, 1)
            alpha = alpha_mask / alpha_sum
            # alpha: (batch_size, 1, 1)
        attn_pool = torch.matmul(alpha, x.transpose(0, 1)).squeeze(1)
        # attn_pool: (batch_size, global_size)
        return attn_pool


class DialogueRNNCell(nn.Module):
    def __init__(self, args):
        super(DialogueRNNCell, self).__init__()
        self.args = args
        self.global_cell = nn.GRUCell(args.feature_size + args.party_size, args.global_size)
        self.party_cell = nn.GRUCell(args.feature_size + args.global_size, args.party_size)
        self.emotion_cell = nn.GRUCell(args.party_size, args.emotion_size)
        self.listener_cell = nn.GRUCell(args.feature_size + args.party_size, args.party_size)
        self.attn = Attention(args.global_size, args.feature_size, attn_type='general')
    
    def forward(self, feature, speaker, last_global_state, last_speaker_state, last_emotion):
        # feature: (batch_size, feature_size)
        # speaker: (batch_size, num_person)
        # last_global_state: (t-1, batch_size, global_size)
        # last_speaker_state: (batch_size, num_person, party_size)
        speaker_idx = torch.argmax(speaker, dim=1)
        # speaker_idx: (batch_size)
        selected_last_speaker_state = []
        for q, idx in zip(last_speaker_state, speaker_idx):
            selected_last_speaker_state.append(q[idx])
        selected_last_speaker_state = torch.stack(selected_last_speaker_state)
        # selected_last_speaker_state: (batch_size, party_size)
        if last_global_state.size(0) == 0:
            global_state = F.dropout(self.global_cell(torch.cat([feature, selected_last_speaker_state], dim=-1), \
                                                torch.zeros(feature.size(0), self.args.global_size).to(feature.device)), \
                        p=self.args.dropout_rate, training=self.training)
            context = torch.zeros(feature.size(0), self.args.global_size).to(feature.device)
        else:
            global_state = F.dropout(self.global_cell(torch.cat([feature, selected_last_speaker_state], dim=-1), last_global_state[-1]), \
                                    p=self.args.dropout_rate, training=self.training)
            context = self.attn(last_global_state, feature)
        # global_state: (batch_size, global_size)
        # context: (batch_size, global_size)
        feature_c = torch.cat([feature, context], dim=-1).unsqueeze(1).expand(-1, speaker.size(1), -1)
        # feature_c: (batch_size, num_person, feature_size + global_size)
        speaker_state = F.dropout(self.party_cell(feature_c.contiguous().view(-1, self.args.feature_size + self.args.global_size), \
                                                        last_speaker_state.view(-1, self.args.party_size)).view(feature.size(0), -1, self.args.party_size), \
                                        p=self.args.dropout_rate, training=self.training)
        # speaker_state: (batch_size, num_person, party_size)

        if self.args.listener_state:
            _feature = feature.unsqueeze(1).expand(-1, speaker.size(1), -1).contiguous().view(-1, self.args.feature_size)
            # _feature: (batch_size * num_person, feature_size)
            selected_speaker_state = []
            for q, idx in zip(speaker_state, speaker_idx):
                selected_speaker_state.append(q[idx])
            selected_speaker_state = torch.stack(selected_speaker_state)
            # selected_speaker_state: (batch_size, party_size)
            selected_speaker_state = selected_speaker_state.unsqueeze(1).expand(-1, speaker.size(1), -1).contiguous().view(-1, self.args.party_size)
            # selected_speaker_state: (batch_size * num_person, party_size)
            listener_state = F.dropout(self.listener_cell(torch.cat([_feature, selected_speaker_state], dim=1), \
                                                        last_speaker_state.view(-1, self.args.party_size)).view(feature.size(0), -1, self.args.party_size), \
                                    p=self.args.dropout_rate, training=self.training)
        else:
            listener_state = last_speaker_state
        
        speaker_state = listener_state * (1 - speaker.unsqueeze(2)) + speaker_state * speaker.unsqueeze(2)
        # speaker_state: (batch_size, num_person, party_size)
        selected_speaker_state = []
        for q, idx in zip(speaker_state, speaker_idx):
            selected_speaker_state.append(q[idx])
        selected_speaker_state = torch.stack(selected_speaker_state)
        # selected_last_speaker_state: (batch_size, party_size)
        if last_emotion.size(0) == 0:
            emotion = self.emotion_cell(selected_speaker_state, torch.zeros(speaker.size(0), self.args.emotion_size).to(selected_speaker_state.device))
        else:
            emotion = self.emotion_cell(selected_speaker_state, last_emotion)
        # emotion: (batch_size, emotion_size)
        emotion = F.dropout(emotion, p=self.args.dropout_rate, training=self.training)
        # emotion: (batch_size, emotion_size)
        return global_state, speaker_state, emotion


class DialogueRNN(nn.Module):
    def __init__(self, args):
        super(DialogueRNN, self).__init__()
        self.args = args
        self.dialogue_rnn_cell = DialogueRNNCell(args)
    
    def forward(self, feature, speaker):
        # feature: (seq_len, batch_size, feature_size)
        # speaker: (seq_len, batch_size, num_person)
        global_state = torch.zeros(0).to(feature.device)
        speaker_state = torch.zeros(speaker.size(1), speaker.size(2), self.args.party_size).to(feature.device)
        # speaker_state: (batch_size, num_person, party_size)
        last_emotion = torch.zeros(0).to(feature.device)
        emotion = last_emotion
        for _feature, _speaker in zip(feature, speaker):
            last_global_state, speaker_state, last_emotion = self.dialogue_rnn_cell(_feature, _speaker, global_state, speaker_state, last_emotion)
            # last_global_state: (batch_size, global_size), speaker_state: (batch_size, num_person, party_size), last_emotion: (batch_size, emotion_size)
            global_state = torch.cat([global_state, last_global_state.unsqueeze(0)], dim=0)
            emotion = torch.cat([emotion, last_emotion.unsqueeze(0)], dim=0)
        # emotion: (seq_len, batch_size, emotion_size)
        return emotion


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
    
    def forward(self, emotion):
        # emotion: (seq_len, batch_size, emotion_size)
        emotion = emotion.transpose(0, 1).contiguous().view(-1, emotion.size(-1))
        # emotion: (batch_size * seq_len, num_class)
        return self.fc(emotion)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dialogue_rnn = DialogueRNN(args)
        self.attn = Attention(args.emotion_size, args.emotion_size, attn_type='general2')
        self.classify = Classifier(args.emotion_size, args.num_class, args.dropout_rate)

    def forward(self, feature, speaker, mask):
        # feature: (seq_len, batch_size, feature_size)
        # speaker: (seq_len, batch_size, num_person)
        # mask: (batch_size, seq_len)
        emotion = self.dialogue_rnn(feature, speaker)
        # emotion: (seq_len, batch_size, emotion_size)
        if self.args.attn:
            attn_emotion = []
            for emo in emotion:
                attn_emo = self.attn(emotion, emo, mask=mask)
                attn_emotion.append(attn_emo)
            attn_emotion = torch.stack(attn_emotion)
            final = self.classify(attn_emotion)
        else:
            final = self.classify(emotion)
        # final: (batch_size * seq_len, num_class)
        return final

class BiModel(nn.Module):
    def __init__(self, args):
        super(BiModel, self).__init__()
        self.args = args
        self.dialogue_rnn_forward = DialogueRNN(args)
        self.dialogue_rnn_reverse = DialogueRNN(args)
        self.attn = Attention(args.emotion_size * 2, args.emotion_size * 2, attn_type='general2')
        self.classify = Classifier(args.emotion_size * 2, args.num_class, args.dropout_rate)
    
    def forward(self, feature, speaker, mask):
        # feature: (seq_len, batch_size, feature_size)
        # speaker: (seq_len, batch_size, num_person)
        # mask: (batch_size, seq_len)
        emotion_forward = F.dropout(self.dialogue_rnn_forward(feature, speaker), p=self.args.dropout_rate + 0.15, training=self.training)
        # emotion_forward: (seq_len, batch_size, emotion_size)
        feature_reverse = pad_sequence([torch.flip(_feature[:_mask], [0]) for _feature, _mask in zip(feature.transpose(0, 1), torch.sum(mask, 1).int())])
        # feature_reverse: (seq_len, batch_size, feature_size)
        speaker_reverse = pad_sequence([torch.flip(_speaker[:_mask], [0]) for _speaker, _mask in zip(speaker.transpose(0, 1), torch.sum(mask, 1).int())])
        # speaker_reverse: (seq_len, batch_size, num_person)
        emotion_reverse = self.dialogue_rnn_reverse(feature_reverse, speaker_reverse)
        emotion_reverse = pad_sequence([torch.flip(_emotion_reverse[:_mask], [0]) for _emotion_reverse, _mask in zip(emotion_reverse.transpose(0, 1), torch.sum(mask, 1).int())])
        emotion_reverse = F.dropout(emotion_reverse, p=self.args.dropout_rate + 0.15, training=self.training)
        # emotion_reverse: (seq_len, batch_size, emotion_size)
        emotion = torch.cat([emotion_forward, emotion_reverse], dim=-1)
        # emotion: (seq_len, batch_size, emotion_size * 2)
        if self.args.attn:
            attn_emotion = []
            for emo in emotion:
                attn_emo = self.attn(emotion, emo, mask=mask)
                attn_emotion.append(attn_emo)
            attn_emotion = torch.stack(attn_emotion)
            final = self.classify(attn_emotion)
        else:
            final = self.classify(emotion)
        # final: (batch_size * seq_len, num_class)
        return final

