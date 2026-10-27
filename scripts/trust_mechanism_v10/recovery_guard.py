"""Paired checkpoint benefit screening. No sequential confidence guarantee."""
import numpy as np

def checkpoint_benefit(current_losses,checkpoint_losses,margin,standard_errors):
    current=np.asarray(current_losses,dtype=float)
    saved=np.asarray(checkpoint_losses,dtype=float)
    if current.ndim!=1 or saved.shape!=current.shape or len(current)<2:
        return False,{'reason':'insufficient_or_unpaired_examples'}
    if not np.isfinite(current).all() or not np.isfinite(saved).all():
        raise ValueError('nonfinite recovery losses')
    differences=current-saved
    gain=float(np.mean(differences))
    se=float(np.std(differences,ddof=1)/np.sqrt(len(differences)))
    return bool(gain>margin+standard_errors*se),{'mean_loss_gain':gain,'standard_error':se,'n':len(current)}
