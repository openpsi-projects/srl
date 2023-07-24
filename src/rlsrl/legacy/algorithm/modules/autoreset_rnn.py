import torch
import torch.nn as nn


class AutoResetRNN(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=1,
                 batch_first=False,
                 rnn_type='lstm'):
        super().__init__()
        self.__type = rnn_type
        if self.__type == 'gru':
            self.__net = nn.GRU(input_dim,
                                output_dim,
                                num_layers=num_layers,
                                batch_first=batch_first)
        elif self.__type == 'lstm':
            self.__net = nn.LSTM(input_dim,
                                 output_dim,
                                 num_layers=num_layers,
                                 batch_first=batch_first)
        else:
            raise NotImplementedError(
                f'RNN type {self.__type} has not been implemented.')

    def __forward(self, x, h):
        if self.__type == 'lstm':
            h = torch.split(h, h.shape[-1] // 2, dim=-1)
            h = (h[0].contiguous(), h[1].contiguous())
        x_, h_ = self.__net(x, h)
        if self.__type == 'lstm':
            h_ = torch.cat(h_, -1)
        return x_, h_

    def forward(self, x, h, on_reset=None):
        if on_reset is None:
            return self.__forward(x, h)

        masks = 1 - on_reset
        hxs = h

        has_zeros = (masks[1:] == 0.0).any(dim=1).nonzero(
            as_tuple=True)[0].cpu().numpy()
        has_zeros = [0] + (has_zeros + 1).tolist() + [x.shape[0]]

        outputs = []
        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            rnn_scores, hxs = self.__forward(
                x[start_idx:end_idx],
                hxs * masks[start_idx].view(1, -1, *((1, ) * (hxs.dim() - 2))))
            outputs.append(rnn_scores)

        # assert len(outputs) == T
        # x is a (T, N, -1) tensor
        x_ = torch.cat(outputs, dim=0)
        h_ = hxs
        return x_, h_
