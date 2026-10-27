"""Paper-based FedVal Eq. 2: label scoring, positive weights, adaptive s2.

Independent PyTorch implementation. The released author code has inconsistent
function signatures and a different adaptive search; see BASELINE_AUDIT.md.
"""
import copy
import numpy as np
import torch
from .research.contracts import AggregationResult
from .research.runner_helpers import _apply_delta


def score_weights(losses, s2, s1=3.0, baseline_factor=3.0):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 2 or not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError('FedVal expects finite nonnegative client-by-label losses')
    means = losses.mean(0)
    deviations = means[None, :] - losses
    mad = np.abs(deviations).mean(0)
    # Identical client losses contribute only Cs1; avoid 0/0 without inventing ranking.
    slopes = np.divide(s1*deviations, mad[None, :],
                       out=np.zeros_like(deviations), where=mad[None, :] > 1e-12)
    ratios = means / max(float(means.mean()), 1e-12)
    bias = np.maximum(1.0, np.maximum(ratios, 1e-12)**s2)
    scores = np.maximum(0, (bias[None, :]*slopes + baseline_factor*s1).sum(1))
    if not np.isfinite(scores).all() or scores.sum() <= 0:
        raise ValueError('FedVal produced no finite positive aggregation weights')
    return scores/scores.sum()


class FedVal:
    def __init__(self, runner, features, labels, outputs):
        self.runner, self.features, self.labels = runner, features, labels
        self.outputs, self.s2 = outputs, 3.0
        if set(np.unique(labels)) != set(range(outputs)):
            raise ValueError('FedVal reference must cover every task label')

    def losses(self, model):
        self.runner.validation_work['reference_example_evaluations'] += len(self.labels)
        self.runner.validation_work['reference_forward_passes'] += (len(self.labels)+self.runner.config.evaluation_batch_size-1)//self.runner.config.evaluation_batch_size
        sums, counts = np.zeros(self.outputs), np.zeros(self.outputs)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(self.labels), self.runner.config.evaluation_batch_size):
                batch = self.runner.config.evaluation_batch_size
                x = torch.as_tensor(self.features[start:start+batch], device=self.runner.device)
                y = torch.as_tensor(self.labels[start:start+batch], device=self.runner.device)
                # Match authors' probability floor in the evaluation loss.
                probabilities = model(x).softmax(1).gather(1, y[:, None]).squeeze(1)
                values = -probabilities.clamp_min(0.0001).log().cpu().numpy()
                yy = y.cpu().numpy()
                sums += np.bincount(yy, weights=values, minlength=self.outputs)
                counts += np.bincount(yy, minlength=self.outputs)
        return sums/counts

    def aggregate(self, model, updates):
        matrix = []
        for update in updates:
            candidate = copy.deepcopy(model)
            _apply_delta(candidate, dict(update.state))
            matrix.append(self.losses(candidate))
        search = [self.s2, self.s2+0.5, self.s2-0.5, self.s2-5, self.s2+5]
        options = []
        for exponent in search:
            weights = score_weights(matrix, exponent)
            state = {key: sum(update.state[key]*float(w) for update, w in zip(updates, weights))
                     if value.is_floating_point() else value.detach().cpu()
                     for key, value in model.state_dict().items()}
            candidate = copy.deepcopy(model)
            _apply_delta(candidate, state)
            options.append((float(self.losses(candidate).mean()), state, weights))
        index = int(np.argmin([option[0] for option in options]))
        self.s2 = search[index]
        _, state, weights = options[index]
        accepted = tuple(u.client_id for u, w in zip(updates, weights) if w > 0)
        rejected = tuple(u.client_id for u, w in zip(updates, weights) if w == 0)
        return AggregationResult(state, accepted, rejected,
            {'s2': self.s2, 'weights': weights.tolist(), 'reference_macro_loss': options[index][0]})
