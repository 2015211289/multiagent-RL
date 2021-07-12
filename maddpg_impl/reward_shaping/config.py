class Config:
    gamma=0.99
    batch_size=32
    lr=0.001
    initial_exploration=1000
    log_interval=10
    update_target=1000
    replay_memory_capacity=1000
    device="cpu"
    sequence_length=32
    burn_in_length=4
    eta=0.9
    local_mini_batch=8
    n_step=1
    over_lapping_length=16
    epsilon_decay=0.00001
    random_seed=42
    enable_ngu=True
    hidden_size=64
    embed_hidden_size = 16
    embed_lr = 1e-4
    beta = 0.0001 
    train_episode_num = 5