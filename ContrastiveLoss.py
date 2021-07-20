from __future__ import print_function

import torch
import torch.nn as nn



class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, beta, add_one_in_neg, exact_cov, exact_cov_unaug_sim):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.beta = beta
        self.temperature = temperature
        self.add_one_in_neg = add_one_in_neg
        self.exact_cov = exact_cov
        self.exact_cov_unaug_sim = exact_cov_unaug_sim
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.mask_samples_small = self._get_correlated_mask_small().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def need_unaug_data(self):
        return self.exact_cov_unaug_sim

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_correlated_mask_small(self):
        diag = np.eye(self.batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, zs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        if self.exact_cov:
            # 1 - sim = dist
            r_neg = 1 - negatives
            r_pos = 1 - positives

            num_negative = negatives.size(1)

            # Similarity matrix for unaugmented data.
            if self.exact_cov_unaug_sim and zs is not None:
                similarity_matrix2 = self.similarity_function(zs, zs)
                negatives_unaug = similarity_matrix2[self.mask_samples_small].view(self.batch_size, -1)
                r_neg_unaug = 1 - negatives_unaug
                w = (-r_neg_unaug.detach() / self.temperature).exp() 
                # Duplicated four times. 
                w = torch.cat([w, w], dim=0)
                w = torch.cat([w, w], dim=1)
            else:
                w = (-r_neg.detach() / self.temperature).exp() 
            
            w = w / (1 + w) / self.temperature / num_negative
            # Then we construct the loss function. 
            w_pos = w.sum(dim=1, keepdim=True)
            loss = (w_pos * r_pos - (w * r_neg).sum(dim=1)).mean()
            loss_intra = self.beta * (w_pos * r_pos).mean()
        else:
            if self.add_one_in_neg:
                all_ones = torch.ones(2 * self.batch_size, 1).to(self.device)
                logits = torch.cat((positives, negatives, all_ones), dim=1)
            else:
                logits = torch.cat((positives, negatives), dim=1)

            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            loss = self.criterion(logits, labels)

            # Make positive strong than negative to trigger an additional term. 
            loss_intra = -positives.sum() * self.beta / self.temperature
            loss /= (1.0 + self.beta) * 2 * self.batch_size
            loss_intra /= (1.0 + self.beta) * 2 * self.batch_size

        return loss, loss_intra