import yaml

class Config:
    def __init__(self, config_path: str='./config.yaml'):
        config = self._load_config(config_path)

        self.train_data_dir = config['train_data_dir']
        self.val_data_dir = config['val_data_dir']
        self.test_data_dir = config['test_data_dir']

        self.tokeniser_dir = config['tokeniser_dir']
        self.deepseek_v3_ckpt_dir = config['deepseek_v3_ckpt_dir']

        tokeniser_config = config['tokeniser_training']
        self.vocab_size = tokeniser_config['vocab_size']
        self.min_frequency = tokeniser_config['min_frequency']
        self.chunk_size = tokeniser_config['chunk_size']
        
        training_config = config['model_training']
        self.max_len = training_config['max_len']
        self.stride = training_config['stride']
        self.batch_size = training_config['batch_size']
        self.num_epochs = training_config['num_epochs']
        self.learning_rate = training_config['learning_rate']
        self.min_learning_rate = training_config['min_learning_rate']
        self.weight_decay = training_config['weight_decay']
        self.mtp_weight = training_config['mtp_weight']
        self.warmup_steps = training_config['warmup_steps']

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config