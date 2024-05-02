_base_ = "./_main_.py"
federated = dict(
    type='apfl',
    personal=True,
    apfl_alpha=0.5, # The alpha variable for the personalized training in APFL algorithm
    apfl_adaptive_alpha=False, # If set, the alpha variable for APFL training will be optimized during training.
    personal_test=False, # If set, the personalized model will be evaluated using test dataset as well.
    apfl_model_avg=False,
)