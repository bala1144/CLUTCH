from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from .utils import *
from bert_score import score as score_bert
import spacy
from mGPT.config import instantiate_from_config
import torch.nn.functional as F

def top_k_retrieval_accuracy(text_embed, motion_embed, k=1):
    """
    Compute Top-K Retrieval Accuracy between text and motion embeddings.
    Args:
        text_embed (Tensor): [B, D] text embeddings
        motion_embed (Tensor): [B, D] motion embeddings
        k (int): Top-K value (e.g., 1, 3)
    Returns:
        accuracy (float): Top-K retrieval accuracy
    """
    # Normalize for cosine similarity
    text_embed = F.normalize(text_embed, dim=-1)
    motion_embed = F.normalize(motion_embed, dim=-1)

    # Compute similarity matrix (B x B)
    sim_matrix = torch.matmul(text_embed, motion_embed.T)

    # Get top-k indices along each row
    topk_indices = sim_matrix.topk(k, dim=-1).indices

    # Ground truth: correct match is on the diagonal
    target = torch.arange(text_embed.size(0)).to(text_embed.device).unsqueeze(1)

    # Compare with top-k predictions
    correct = (topk_indices == target).any(dim=1).float()

    # Mean accuracy
    return correct.mean()

class TM2TMetrics_V2(Metric):

    def __init__(self,
                 cfg,
                 w_vectorizer,
                 dataname='humanml3d',
                 top_k=3,
                 bleu_k=4,
                 R_size=32,
                 max_text_len=40,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 unit_length=4,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.R_size = R_size

        self.cfg = cfg
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = [ "top_1_retrieval_accuracy", "top_3_retrieval_accuracy"]
        self.add_state("top_1_retrieval_accuracy",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        
        self.add_state("top_3_retrieval_accuracy",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")

        self.add_state("text_embeddings", default=[])
        self.add_state("motion_embeddings", default=[])

        ### self.matching_metrics
        self.add_state("Matching_score",
                            default=torch.tensor(0.0),
                            dist_reduce_fx="sum")
        self.Matching_metrics = ["Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        self.metrics.extend(self.Matching_metrics)

    @torch.no_grad()
    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()
        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics
        
        shuffle_idx = torch.randperm(count_seq)
        all_motions = torch.cat(self.motion_embeddings,axis=0).cpu()[shuffle_idx, :]
        all_texts = torch.cat(self.text_embeddings,axis=0).cpu()[shuffle_idx, :]
        
        metrics["top_1_retrieval_accuracy"] =  top_k_retrieval_accuracy(all_texts, all_motions, k=1)
        metrics["top_3_retrieval_accuracy"] =  top_k_retrieval_accuracy(all_texts, all_motions, k=3)


        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_motions[i * self.R_size:(i + 1) *
                                            self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(
                group_texts, group_motions).nan_to_num()
            # print(dist_mat[:5])
            self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax,
                                            top_k=self.top_k).sum(axis=0)

        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = self.Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count
            

        # Reset
        self.reset()
        return {**metrics}

    @torch.no_grad()
    def update(self,
               motion_embed: Tensor,
               text_embed: Tensor,
               lengths: List[int],
               word_embs: Tensor = None,
               pos_ohot: Tensor = None,
               text_lengths: Tensor = None):

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        text_embeddings = torch.flatten(text_embed, start_dim=1).detach()
        motion_embeddings = torch.flatten(motion_embed, start_dim=1).detach()
        self.text_embeddings.append(text_embeddings)
        self.motion_embeddings.append(motion_embeddings)

        # # motion encoder
        # m_lens = torch.tensor(lengths, device=feats_ref.device)
        # align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        # feats_ref = feats_ref[align_idx]
        # m_lens = m_lens[align_idx]
        # m_lens = torch.div(m_lens,
        #                    self.cfg.DATASET.HUMANML3D.UNIT_LEN,
        #                    rounding_mode="floor")
        # ref_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        # m_lens = m_lens // self.unit_length
        # ref_emb = self.t2m_motionencoder(ref_mov, m_lens)

        # self.gtmotion_embeddings.append(gtmotion_embeddings)

        # # text encoder
        # gttext_emb = self.t2m_textencoder(word_embs, pos_ohot,
        #                                   text_lengths)[align_idx]

        # predtext_emb = self._get_text_embeddings(pred_texts)[align_idx]
        # predtext_embeddings = torch.flatten(predtext_emb, start_dim=1).detach()