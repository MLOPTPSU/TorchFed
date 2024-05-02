data=dict(
    data_dir='./data/',
    pin_memory=True,
    num_workers=4,
    # The next two parameters are used for the growing batch size training.
    base_batch_size=1,
    max_batch_size=0,
)