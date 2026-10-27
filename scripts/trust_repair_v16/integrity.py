"""Independent CPU arithmetic and exact model-state checks at update boundaries.

These checks detect inconsistencies; they do not repair weights or identify the
underlying hardware/runtime cause. Exact mismatches stop the experiment.
"""
import hashlib
import numpy as np
import torch


class IntegrityError(RuntimeError):
    def __init__(self,message,**evidence):
        super().__init__(message)
        self.evidence=evidence


def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def state_hash(state):
    digest=hashlib.sha256()
    for value in state.values():
        array=np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(str((array.shape,array.dtype)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def assert_state(model,expected,context):
    actual=snapshot(model)
    if set(actual)!=set(expected):
        raise IntegrityError(f'{context}: model keys changed',expected=expected,actual=actual)
    differences={k:int((actual[k]!=value).sum()) if actual[k].shape==value.shape else -1
                 for k,value in expected.items() if not torch.equal(actual[k],value)}
    if differences:
        raise IntegrityError(f'{context}: model state mismatch {differences}',
                             expected=expected,actual=actual,differences=differences)
    return state_hash(actual)


def assert_hash(model,expected,context):
    actual=snapshot(model)
    observed=state_hash(actual)
    if observed!=expected:
        raise IntegrityError(f'{context}: model digest changed',expected_hash=expected,
                             actual_hash=observed,actual=actual)
    return observed


def expected_state(parent,delta):
    expected={}
    for name,value in parent.items():
        if value.is_floating_point():
            update=delta[name].detach().cpu().clone()
            if update.shape!=value.shape or update.dtype!=value.dtype:
                raise IntegrityError('update shape/dtype mismatch',parameter=name)
            candidate=value+update
            if not torch.isfinite(candidate).all():
                raise IntegrityError('nonfinite expected model',parameter=name,parent=parent,delta=delta)
            expected[name]=candidate
        else:
            expected[name]=value.clone()
    return expected


def checked_apply_delta(model,delta):
    parent=snapshot(model)
    expected=expected_state(parent,delta)
    # Perform arithmetic on CPU. The previous GPU add/load path produced a
    # one-element corruption in 7.weight on an RTX 2080 Ti.
    state=model.state_dict()
    updated={name:expected[name].to(value.device,dtype=value.dtype) if value.is_floating_point() else value
             for name,value in state.items()}
    model.load_state_dict(updated)
    assert_state(model,expected,'apply_delta')
