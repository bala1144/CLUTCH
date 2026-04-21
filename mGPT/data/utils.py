import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float().clone().detach() for b in batch]
    else:
        batch = [b.float().clone().detach() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def humanml3d_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float().clone().detach()  for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float().clone().detach() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float().clone().detach() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) == 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    if len(notnone_batches[0]) == 10:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})
        adapted_batch.update({"seq_name": [b[9] for b in notnone_batches]})


    return adapted_batch

from torch.utils.data._utils.collate import default_collate

def grab_collate(batch):

    if type(batch[0]) == dict : ## list of dict

        out_batch_dict = {}
        all_keys = list(batch[0].keys())
        for k in all_keys:
            if k in ["motion", "word_embs", "pos_ohot", "text_len", "motion_ref", "motion_mask", "sentence_vec"]:
                sampl_ = [torch.tensor(b[k]).float().clone().detach() for b in batch]
                out_batch_dict[k] = collate_tensors(sampl_)
            else:
                out_batch_dict[k] = [b[k] for b in batch]
                
        return out_batch_dict


    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) == 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    return adapted_batch



# def grab_collate(batch):

#     if type(batch[0]) == dict : ## list of dict
#         out_batch_dict = default_collate(batch)
#         if "length" in out_batch_dict.keys():
#             out_batch_dict["length"] = out_batch_dict["length"].cpu().tolist()

#         if "tasks" in out_batch_dict.keys():
#             out_batch_dict["tasks"] = [b["tasks"] for b in batch]
        
#         return out_batch_dict


#     notnone_batches = [b for b in batch if b is not None]
#     EvalFlag = False if notnone_batches[0][5] is None else True

#     # Sort by text length
#     if EvalFlag:
#         notnone_batches.sort(key=lambda x: x[5], reverse=True)

#     # Motion only
#     adapted_batch = {
#         "motion":
#         collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
#         "length": [b[2] for b in notnone_batches],
#     }

#     # Text and motion
#     if notnone_batches[0][0] is not None:
#         adapted_batch.update({
#             "text": [b[0] for b in notnone_batches],
#             "all_captions": [b[7] for b in notnone_batches],
#         })

#     # Evaluation related
#     if EvalFlag:
#         adapted_batch.update({
#             "text": [b[0] for b in notnone_batches],
#             "word_embs":
#             collate_tensors(
#                 [torch.tensor(b[3]).float() for b in notnone_batches]),
#             "pos_ohot":
#             collate_tensors(
#                 [torch.tensor(b[4]).float() for b in notnone_batches]),
#             "text_len":
#             collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
#             "tokens": [b[6] for b in notnone_batches],
#         })

#     # Tasks
#     if len(notnone_batches[0]) == 9:
#         adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

#     return adapted_batch

def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
