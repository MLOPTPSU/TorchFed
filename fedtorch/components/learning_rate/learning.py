# -*- coding: utf-8 -*-
from ..scheduler_builder import LR_SCHEDULER

"""the entry to init the lr."""

@LR_SCHEDULER.register_module()
def strict(lr_change_epochs, lr_fields, lr_scale_indicators, num_epochs):
    # define lr_fields
    lr_change_epochs = '0,{original},{full}'.format(
        original=lr_change_epochs, full=num_epochs
    )
    epoch_fields, lr_fields, scale_indicators = _get_scheduling_setup(lr_change_epochs, lr_fields, lr_scale_indicators)
    lr_schedulers = _build_lr_schedulers(epoch_fields, lr_fields, scale_indicators) 
    return _get_lr_scheduler(epoch_fields, lr_schedulers)

@LR_SCHEDULER.register_module()
def onecycle(lr_onecycle_low, 
            lr_onecycle_high, 
            lr_onecycle_extra_low,
            lr_onecycle_num_epoch, 
            num_epochs):
    lr_fields = '{low},{high}/{high},{low}/{low},{extra_low}'.format(
        low=lr_onecycle_low,
        high=lr_onecycle_high,
        extra_low=lr_onecycle_extra_low
    )
    lr_change_epochs = '0,{half_cycle},{cycle},{full}'.format(
        half_cycle=lr_onecycle_num_epoch // 2,
        cycle=lr_onecycle_num_epoch,
        full=num_epochs
    )
    lr_scale_indicators = '0,0,0'
    epoch_fields, lr_fields, scale_indicators = _get_scheduling_setup(lr_change_epochs, lr_fields, lr_scale_indicators)
    lr_schedulers = _build_lr_schedulers(epoch_fields, lr_fields, scale_indicators) 
    return _get_lr_scheduler(epoch_fields, lr_schedulers)

@LR_SCHEDULER.register_module()
def multistep(lr_change_epochs,
              lr_warmup, 
              learning_rate, 
              init_warmup_lr, 
              lr_decay, 
              num_epochs,
              lr_warmup_epochs):
    # define lr_fields
    lr_fields = _build_multistep_lr_fields(
        lr_change_epochs,
        lr_warmup, learning_rate, init_warmup_lr, lr_decay)

    # define lr_change_epochs
    lr_change_epochs, num_intervals = _build_multistep_lr_change_epochs(
        lr_change_epochs, lr_warmup, lr_warmup_epochs,
        num_epochs)

    # define scale_indicators
    lr_scale_indicators = ','.join(['0'] * num_intervals)
    epoch_fields, lr_fields, scale_indicators = _get_scheduling_setup(lr_change_epochs, lr_fields, lr_scale_indicators)
    lr_schedulers = _build_lr_schedulers(epoch_fields, lr_fields, scale_indicators) 
    return _get_lr_scheduler(epoch_fields, lr_schedulers)

@LR_SCHEDULER.register_module()
def convex_decay(learning_rate,
                 num_epochs,
                 lr_gamma,
                 lr_mu,
                 lr_alpha):
    # define lr_fields
    lr_fields = '{},{}'.format(learning_rate, 0)

    # define lr_change_epochs
    lr_change_epochs = '0,{full}'.format(full=num_epochs)

    # define scale_indicators
    lr_scale_indicators = '2'
    epoch_fields, lr_fields, scale_indicators = _get_scheduling_setup(lr_change_epochs, lr_fields, lr_scale_indicators)
    lr_schedulers = _build_lr_schedulers(epoch_fields, lr_fields, scale_indicators,
                                         lr_gamma=lr_gamma,lr_mu=lr_mu,lr_alpha=lr_alpha)
    return _get_lr_scheduler(epoch_fields, lr_schedulers)


def _build_lr_schedulers(epoch_fields, lr_fields, scale_indicators, **kwargs):
    lr_schedulers = dict()

    for field_id, (epoch_field, lr_field, indicator) in \
            enumerate(zip(epoch_fields, lr_fields, scale_indicators)):
        lr_scheduler = _build_lr_scheduler(epoch_field, lr_field, indicator, **kwargs)
        lr_schedulers[field_id] = lr_scheduler
    return lr_schedulers


def _build_lr_scheduler(epoch_field, lr_field, scale_indicator, **kwargs):
    lr_left, lr_right = lr_field
    epoch_left, epoch_right = epoch_field
    n_steps = epoch_right - epoch_left

    if scale_indicator == 'linear':
        return _linear_scale(lr_left, lr_right, n_steps, epoch_left)
    elif scale_indicator == 'poly':
        return _poly_scale(lr_left, lr_right, n_steps, epoch_left)
    elif scale_indicator == 'convex':
        assert kwargs['lr_gamma'] is not None
        assert kwargs['lr_mu'] is not None
        assert kwargs['lr_alpha'] is not None
        return _convex_scale(kwargs['lr_gamma'], kwargs['lr_mu'], kwargs['lr_alpha'])
    else:
        raise NotImplementedError


def _get_lr_scheduler(epoch_fields, lr_schedulers):
    def f(epoch_index):
        return _get_lr_scheduler_fn(epoch_index, epoch_fields, lr_schedulers)
    return f


def _get_lr_scheduler_fn(epoch_index, epoch_fields, lr_schedulers):
    """Note that epoch index is a floating number."""
    def _is_fall_in(index, left_index, right_index):
        return left_index <= index < right_index

    for ind, (epoch_left, epoch_right) in enumerate(epoch_fields):
        if _is_fall_in(epoch_index, epoch_left, epoch_right):
            return lr_schedulers[ind](epoch_index)


"""Define the scheduling step,
    e.g., logic of epoch_fields, lr_fields and scale_indicators.

    We should be able to determine if we only use the pure info from parser,
    or use a mixed version (the second one might be more common in practice)

    For epoch_fields, we define it by a string separated by ',',
    e.g., '10,20,30' to indicate different ranges. more precisely,
    previous example is equivalent to three different ranges [0, 10), [10, 20), [20, 30).

    For scale_indicators,
"""


def _get_scheduling_setup(lr_change_epochs, lr_fields, lr_scale_indicators):
    assert lr_change_epochs is not None
    assert lr_fields is not None
    assert lr_scale_indicators is not None

    # define lr_fields
    lr_fields = _get_lr_fields(lr_fields)

    # define scale_indicators
    scale_indicators = _get_lr_scale_indicators(lr_scale_indicators)

    # define epoch_fields
    epoch_fields = _get_lr_epoch_fields(lr_change_epochs)

    return epoch_fields, lr_fields, scale_indicators


def _build_multistep_lr_fields(
        lr_change_epochs, lr_warmup, learning_rate, init_warmup_lr, lr_decay):
    if lr_change_epochs is not None:
        _lr_fields = [
            learning_rate * ((1. / lr_decay) ** l)
            for l in range(len(lr_change_epochs.split(',')) + 1)
        ]
    else:
        _lr_fields = [learning_rate]

    lr_fields = '/'.join(['{lr},{lr}'.format(lr=lr) for lr in _lr_fields])

    if lr_warmup:
        return '{},{}/'.format(init_warmup_lr, learning_rate) + lr_fields
    else:
        return lr_fields


def _build_multistep_lr_change_epochs(
        lr_change_epochs, lr_warmup, lr_warmup_epochs, num_epochs):
    if lr_change_epochs is not None:
        lr_change_epochs = [0] + lr_change_epochs.split(',') + [num_epochs]
    else:
        lr_change_epochs = [0, num_epochs]

    if lr_warmup:
        lr_change_epochs = [0, lr_warmup_epochs] + lr_change_epochs[1:]
    return ','.join([str(x) for x in lr_change_epochs]), len(lr_change_epochs) - 1




def _get_lr_fields(lr_fields):
    return [map(float, l.split(',')) for l in lr_fields.split('/')]


def _get_lr_scale_indicators(lr_scale_indicators):
    def digital2name(x):
        return {
            '0': 'linear',
            '1': 'poly',
            '2': 'convex'  # lr = \gamma / (\mu (t + a))
        }[x]
    return [digital2name(l) for l in lr_scale_indicators.split(',')]


def _get_lr_epoch_fields(lr_change_epochs):
    """note that the change points exclude the head and tail of the epochs.
    """
    lr_change_epochs = [int(l) for l in lr_change_epochs.split(',')]
    from_s = lr_change_epochs[:-1]
    to_s = lr_change_epochs[1:]
    return list(zip(from_s, to_s))


"""define the learning rate scheduler and the fundamental logic."""


def _linear_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        step = (lr_right - lr_left) / n_steps
        return (index - abs_index) * step + lr_left
    return f


def _poly_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        return lr_left * ((1 - (index - abs_index) / n_steps) ** 2)
    return f


def _convex_scale(gamma, mu, alpha):
    # it is expected in the form of lr = \gamma / (\mu (t + a))
    def f(index):
        return gamma / (mu * (alpha + index))
    return f

