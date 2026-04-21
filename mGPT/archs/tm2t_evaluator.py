import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from mGPT.config import instantiate_from_config

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionTextContrastiveModel(nn.Module):
    def __init__(self, t2m_textencoder, t2m_moveencoder, t2m_motionencoder, **kwargs):
        super(MotionTextContrastiveModel, self).__init__()
        self.t2m_textencoder = instantiate_from_config(t2m_textencoder)
        self.t2m_moveencoder = instantiate_from_config(t2m_moveencoder)
        self.t2m_motionencoder = instantiate_from_config(t2m_motionencoder)

    def contrastive_loss(self, s, p, y, margin=10.0):
        """
        Contrastive loss function for text-motion matching.
        s: text embeddings [B, D]
        p: motion embeddings [B, D]
        y: labels (0 = positive/matched, 1 = negative/mismatched)
        """
        dist_squared = torch.sum((s - p) ** 2, dim=1)
        positive_loss = (1 - y) * dist_squared ** 2
        margin_diff = torch.clamp(margin - dist_squared, min=0.0)
        negative_loss = y * margin_diff ** 2
        return (positive_loss + negative_loss).mean()

    def generate_labels(self, batch_size):
        """
        Create binary labels: 0 for positive (first half), 1 for negative (second half)
        """
        labels = torch.zeros(batch_size, dtype=torch.float32)
        labels[batch_size // 2:] = 1
        return labels

    def get_contrastive_batch(self, motion_inputs, motion_lens, word_embs, pos_ohot, text_lengths):
        """
        Shuffles text encodings for negative examples.
        Returns: motion_inputs, motion_lens, shuffled word_embs, pos_ohot, text_lengths, labels
        """
        batch_size = motion_inputs.size(0)
        labels = self.generate_labels(batch_size).to(motion_inputs.device)

        # Shuffle only second half of text batch
        perm = torch.randperm(batch_size)
        shuffled_word_embs = word_embs.clone()
        shuffled_pos_ohot = pos_ohot.clone()
        shuffled_text_lengths = text_lengths.clone()

        half = batch_size // 2
        shuffled_word_embs[half:] = word_embs[perm][half:]
        shuffled_pos_ohot[half:] = pos_ohot[perm][half:]
        shuffled_text_lengths[half:] = text_lengths[perm][half:]

        return motion_inputs, motion_lens, shuffled_word_embs, shuffled_pos_ohot, shuffled_text_lengths, labels

    def forward(self, motion_inputs, motion_lens, word_embs, pos_ohot, text_lengths):
        # Get augmented batch and labels
        motion_inputs, motion_lens, word_embs, pos_ohot, text_lengths, labels = self.get_contrastive_batch(
            motion_inputs, motion_lens, word_embs, pos_ohot, text_lengths
        )

        movement_feat = self.t2m_moveencoder(motion_inputs)
        motion_embed = self.t2m_motionencoder(movement_feat, motion_lens)  # B x D
        text_embed = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)  # B x D

        loss = self.contrastive_loss(text_embed, motion_embed, labels)

        return loss, motion_embed, text_embed



################################################
################################################
################################################

class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        # self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoderBiGRUCo, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        # self.input_emb.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        # cap_lens = m_lens.data.tolist()
        # emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True)

        emb = input_embs
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(TextEncoderBiGRUCo, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        # self.input_emb.apply(init_weight)
        # self.pos_emb.apply(init_weight)
        # self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        # emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True)
        emb = pack_padded_sequence(input=input_embs, lengths=cap_lens, batch_first=True, enforce_sorted=False)


        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)
