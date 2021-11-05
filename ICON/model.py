import torch
import torch.nn as nn
import torch.nn.functional as F


class ICON(nn.Module):
    def __init__(self, args):
        super(ICON, self).__init__()
        self.args = args

        self.fc = nn.Linear(args.feature_size, args.embedding_size)
        self.local_own_GRU = nn.GRUCell(args.feature_size, args.embedding_size)
        self.local_other_GRU = nn.GRUCell(args.feature_size, args.embedding_size)
        self.global_GRU = nn.GRU(args.embedding_size, args.embedding_size, batch_first=True)
        self.memory_GRU = nn.GRU(args.embedding_size, args.embedding_size, batch_first=True)
        self.classify = nn.Linear(args.embedding_size, args.num_class)
    
    def forward(self, query, own_history, other_history, own_history_mask, other_history_mask):
        assert own_history.size(1) == other_history.size(1) == own_history_mask.size(1) == other_history_mask.size(1)
        batch_size = query.size(0)
        time_step = own_history.size(1)

        # Fusion fully-connected
        query = F.tanh(self.fc(query))

        # SIM
        hidden_vector = torch.zeros(batch_size, self.args.embedding_size).to(own_history.device)
        own_history_rnn_output = []
        for i in range(time_step):
            prev_hidden_vector = hidden_vector
            hidden_vector = self.local_own_GRU(own_history[:, i, :], hidden_vector)
            local_mask = own_history_mask[:, i].unsqueeze(1).repeat(1, self.args.embedding_size)
            hidden_vector = torch.where(local_mask, hidden_vector, prev_hidden_vector)
            own_history_rnn_output.append(torch.where(local_mask, hidden_vector, torch.zeros(batch_size, self.args.embedding_size).to(hidden_vector.device)))
        own_history_rnn_output = F.dropout(torch.stack(own_history_rnn_output).transpose(0, 1), p=self.args.dropout_rate, training=self.training)  # (batch_size, time_step, embedding_size)

        hidden_vector = torch.zeros(batch_size, self.args.embedding_size).to(other_history.device)
        other_history_rnn_output = []
        for i in range(time_step):
            prev_hidden_vector = hidden_vector
            hidden_vector = self.local_other_GRU(other_history[:, i, :], hidden_vector)
            local_mask = other_history_mask[:, i].unsqueeze(1).repeat(1, self.args.embedding_size)
            hidden_vector = torch.where(local_mask, hidden_vector, prev_hidden_vector)
            other_history_rnn_output.append(torch.where(local_mask, hidden_vector, torch.zeros(batch_size, self.args.embedding_size).to(hidden_vector.device)))
        other_history_rnn_output = F.dropout(torch.stack(other_history_rnn_output).transpose(0, 1), p=self.args.dropout_rate, training=self.training)  # (batch_size, time_step, embedding_size)

        # DGIM
        mask = own_history_mask + other_history_mask
        global_GRU_input = F.tanh(own_history_rnn_output + other_history_rnn_output)

        for hop in range(self.args.hop_size):
            if hop == 0:
                rnn_output, _ = self.global_GRU(global_GRU_input, None)
            else:
                rnn_output, _ = self.memory_GRU(rnn_output, None)
            
            rnn_outputs = []
            for j in range(time_step):
                local_mask = mask[:, j].unsqueeze(1).repeat(1, self.args.embedding_size)
                local_output = rnn_output[:, j, :]  # (batch_size, embedding_size)
                output_vector = torch.where(local_mask, local_output, torch.zeros(batch_size, self.args.embedding_size).to(local_output.device))  # (batch_size, embedding_size)
                rnn_outputs.append(output_vector)
            rnn_output = torch.stack(rnn_outputs).transpose(0, 1)  # (batch_size, time_step, embedding_size)

            attn_score = F.tanh(torch.matmul(query.unsqueeze(1), rnn_output.transpose(1, 2)).squeeze())  # (batch_size, 1, embedding_size)  X (batch_size, embedding_size, time_steps) -> (batch_size, time_steps)
            attn_score = torch.where(mask, attn_score, torch.full((batch_size, time_step), float('-inf')).to(attn_score.device))
            attn_score = F.dropout(F.softmax(attn_score), p=self.args.dropout_rate, training=self.training)  # (batch_size, time_step)
            attn_score = torch.where(mask, attn_score, torch.zeros_like(attn_score).to(attn_score.device))
            weight = torch.matmul(attn_score.unsqueeze(1), rnn_output).squeeze()
            query = F.tanh(query + weight)

        # Classify
        final = self.classify(query)

        return final

