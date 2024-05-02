_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='rcv1',
        root="./data/rcv1", 
        num_classes=2,
        dimension=[47236],
        )
)