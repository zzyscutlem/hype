"""
Configuration management for HyPE system.

This module handles loading and validation of system configuration
from YAML/JSON files.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for language models."""
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Value Model specific
    value_head_hidden_dim: int = 4096
    
    # LoRA specific
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training specific
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 3
    warmup_steps: int = 100
    
    def validate(self) -> List[str]:
        """
        Validate model configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.base_model_name:
            errors.append("base_model_name cannot be empty")
        
        if self.device not in ["cuda", "cpu", "mps"]:
            errors.append(f"Invalid device: {self.device}. Must be 'cuda', 'cpu', or 'mps'")
        
        if self.max_length <= 0:
            errors.append(f"max_length must be positive, got {self.max_length}")
        
        if not 0.0 <= self.temperature <= 2.0:
            errors.append(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        
        if not 0.0 <= self.top_p <= 1.0:
            errors.append(f"top_p must be in [0.0, 1.0], got {self.top_p}")
        
        if self.value_head_hidden_dim <= 0:
            errors.append(f"value_head_hidden_dim must be positive, got {self.value_head_hidden_dim}")
        
        if self.lora_rank <= 0:
            errors.append(f"lora_rank must be positive, got {self.lora_rank}")
        
        if self.lora_alpha <= 0:
            errors.append(f"lora_alpha must be positive, got {self.lora_alpha}")
        
        if not 0.0 <= self.lora_dropout < 1.0:
            errors.append(f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}")
        
        if not self.lora_target_modules:
            errors.append("lora_target_modules cannot be empty")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        return errors


@dataclass
class PrincipleMemoryConfig:
    """Configuration for Principle Memory."""
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "principles"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    
    # Milvus Lite (embedded mode) configuration
    use_milvus_lite: bool = False  # Use embedded Milvus instead of server
    milvus_lite_path: str = "./data/milvus_lite.db"  # Local database file path
    
    # Retrieval parameters
    top_k: int = 5
    semantic_weight: float = 0.7  # α in retrieval formula
    
    # Deduplication parameters
    duplicate_threshold: float = 0.85
    merge_threshold: float = 0.75
    
    # Pruning parameters
    min_credit_score: float = 0.1
    min_application_count: int = 3
    max_principles: int = 100000
    
    def validate(self) -> List[str]:
        """
        Validate principle memory configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Only validate server connection params if not using Milvus Lite
        if not self.use_milvus_lite:
            if not self.milvus_host:
                errors.append("milvus_host cannot be empty when use_milvus_lite is False")
            
            if not 1 <= self.milvus_port <= 65535:
                errors.append(f"milvus_port must be in [1, 65535], got {self.milvus_port}")
        else:
            # Validate Milvus Lite path
            if not self.milvus_lite_path:
                errors.append("milvus_lite_path cannot be empty when use_milvus_lite is True")
        
        if not self.collection_name:
            errors.append("collection_name cannot be empty")
        
        if not self.embedding_model:
            errors.append("embedding_model cannot be empty")
        
        if self.embedding_dim <= 0:
            errors.append(f"embedding_dim must be positive, got {self.embedding_dim}")
        
        if self.top_k <= 0:
            errors.append(f"top_k must be positive, got {self.top_k}")
        
        if not 0.0 <= self.semantic_weight <= 1.0:
            errors.append(f"semantic_weight must be in [0.0, 1.0], got {self.semantic_weight}")
        
        if not 0.0 <= self.duplicate_threshold <= 1.0:
            errors.append(f"duplicate_threshold must be in [0.0, 1.0], got {self.duplicate_threshold}")
        
        if not 0.0 <= self.merge_threshold <= 1.0:
            errors.append(f"merge_threshold must be in [0.0, 1.0], got {self.merge_threshold}")
        
        if self.min_credit_score < 0:
            errors.append(f"min_credit_score must be non-negative, got {self.min_credit_score}")
        
        if self.min_application_count < 0:
            errors.append(f"min_application_count must be non-negative, got {self.min_application_count}")
        
        if self.max_principles <= 0:
            errors.append(f"max_principles must be positive, got {self.max_principles}")
        
        return errors


@dataclass
class HMCTSConfig:
    """Configuration for H-MCTS planner."""
    search_budget: int = 100
    exploration_constant: float = 1.414  # c in UCB formula
    max_depth: int = 10
    num_hypotheses_per_node: int = 3
    early_stop_threshold: float = 0.8  # Q-value threshold for early stopping
    min_iterations: int = 10  # Minimum iterations before early stopping
    value_cache_size: int = 100  # Cache size for value predictions
    
    def validate(self) -> List[str]:
        """
        Validate H-MCTS configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.search_budget <= 0:
            errors.append(f"search_budget must be positive, got {self.search_budget}")
        
        if self.exploration_constant < 0:
            errors.append(f"exploration_constant must be non-negative, got {self.exploration_constant}")
        
        if self.max_depth <= 0:
            errors.append(f"max_depth must be positive, got {self.max_depth}")
        
        if self.num_hypotheses_per_node <= 0:
            errors.append(f"num_hypotheses_per_node must be positive, got {self.num_hypotheses_per_node}")
        
        if not 0.0 <= self.early_stop_threshold <= 1.0:
            errors.append(f"early_stop_threshold must be in [0.0, 1.0], got {self.early_stop_threshold}")
        
        if self.min_iterations < 0:
            errors.append(f"min_iterations must be non-negative, got {self.min_iterations}")
        
        if self.value_cache_size <= 0:
            errors.append(f"value_cache_size must be positive, got {self.value_cache_size}")
        
        return errors


@dataclass
class SRAdaptConfig:
    """Configuration for SR-Adapt validator."""
    alignment_threshold: float = 0.6
    correction_steps: int = 5
    correction_learning_rate: float = 1e-4
    max_correction_attempts: int = 3
    
    def validate(self) -> List[str]:
        """
        Validate SR-Adapt configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not 0.0 <= self.alignment_threshold <= 1.0:
            errors.append(f"alignment_threshold must be in [0.0, 1.0], got {self.alignment_threshold}")
        
        if self.correction_steps <= 0:
            errors.append(f"correction_steps must be positive, got {self.correction_steps}")
        
        if self.correction_learning_rate <= 0:
            errors.append(f"correction_learning_rate must be positive, got {self.correction_learning_rate}")
        
        if self.max_correction_attempts <= 0:
            errors.append(f"max_correction_attempts must be positive, got {self.max_correction_attempts}")
        
        return errors


@dataclass
class DPVDConfig:
    """Configuration for DPVD learner."""
    principle_weight: float = 0.1  # β in dense reward formula
    discount_factor: float = 0.99
    replay_buffer_size: int = 10000
    training_trigger_threshold: int = 1000
    value_training_epochs: int = 3
    
    def validate(self) -> List[str]:
        """
        Validate DPVD configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.principle_weight < 0:
            errors.append(f"principle_weight must be non-negative, got {self.principle_weight}")
        
        if not 0.0 <= self.discount_factor <= 1.0:
            errors.append(f"discount_factor must be in [0.0, 1.0], got {self.discount_factor}")
        
        if self.replay_buffer_size <= 0:
            errors.append(f"replay_buffer_size must be positive, got {self.replay_buffer_size}")
        
        if self.training_trigger_threshold <= 0:
            errors.append(f"training_trigger_threshold must be positive, got {self.training_trigger_threshold}")
        
        if self.training_trigger_threshold > self.replay_buffer_size:
            errors.append(f"training_trigger_threshold ({self.training_trigger_threshold}) cannot exceed replay_buffer_size ({self.replay_buffer_size})")
        
        if self.value_training_epochs <= 0:
            errors.append(f"value_training_epochs must be positive, got {self.value_training_epochs}")
        
        return errors


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1  # KL penalty coefficient
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 3
    
    def validate(self) -> List[str]:
        """
        Validate DPO configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.beta <= 0:
            errors.append(f"beta must be positive, got {self.beta}")
        
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")
        
        return errors


@dataclass
class HyPEConfig:
    """Main configuration for HyPE system."""
    model: ModelConfig = field(default_factory=ModelConfig)
    principle_memory: PrincipleMemoryConfig = field(default_factory=PrincipleMemoryConfig)
    hmcts: HMCTSConfig = field(default_factory=HMCTSConfig)
    sr_adapt: SRAdaptConfig = field(default_factory=SRAdaptConfig)
    dpvd: DPVDConfig = field(default_factory=DPVDConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    
    # General settings
    seed: int = 42
    log_level: str = "INFO"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'HyPEConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            HyPEConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        
        config = cls.from_dict(config_dict)
        config.validate()
        return config
    
    @classmethod
    def from_json(cls, path: str) -> 'HyPEConfig':
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            HyPEConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        config = cls.from_dict(config_dict)
        config.validate()
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyPEConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            HyPEConfig instance
        """
        model_config = ModelConfig(**config_dict.get('model', {}))
        memory_config = PrincipleMemoryConfig(**config_dict.get('principle_memory', {}))
        hmcts_config = HMCTSConfig(**config_dict.get('hmcts', {}))
        sr_adapt_config = SRAdaptConfig(**config_dict.get('sr_adapt', {}))
        dpvd_config = DPVDConfig(**config_dict.get('dpvd', {}))
        dpo_config = DPOConfig(**config_dict.get('dpo', {}))
        
        return cls(
            model=model_config,
            principle_memory=memory_config,
            hmcts=hmcts_config,
            sr_adapt=sr_adapt_config,
            dpvd=dpvd_config,
            dpo=dpo_config,
            seed=config_dict.get('seed', 42),
            log_level=config_dict.get('log_level', 'INFO'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
            log_dir=config_dict.get('log_dir', './logs'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'principle_memory': self.principle_memory.__dict__,
            'hmcts': self.hmcts.__dict__,
            'sr_adapt': self.sr_adapt.__dict__,
            'dpvd': self.dpvd.__dict__,
            'dpo': self.dpo.__dict__,
            'seed': self.seed,
            'log_level': self.log_level,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
        }
    
    def save_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save JSON file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self):
        """
        Validate all configuration sections.
        
        Raises:
            ValueError: If any validation errors are found
        """
        all_errors = []
        
        # Validate each config section
        model_errors = self.model.validate()
        if model_errors:
            all_errors.extend([f"model.{e}" for e in model_errors])
        
        memory_errors = self.principle_memory.validate()
        if memory_errors:
            all_errors.extend([f"principle_memory.{e}" for e in memory_errors])
        
        hmcts_errors = self.hmcts.validate()
        if hmcts_errors:
            all_errors.extend([f"hmcts.{e}" for e in hmcts_errors])
        
        sr_adapt_errors = self.sr_adapt.validate()
        if sr_adapt_errors:
            all_errors.extend([f"sr_adapt.{e}" for e in sr_adapt_errors])
        
        dpvd_errors = self.dpvd.validate()
        if dpvd_errors:
            all_errors.extend([f"dpvd.{e}" for e in dpvd_errors])
        
        dpo_errors = self.dpo.validate()
        if dpo_errors:
            all_errors.extend([f"dpo.{e}" for e in dpo_errors])
        
        # Validate general settings
        if self.seed < 0:
            all_errors.append(f"seed must be non-negative, got {self.seed}")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            all_errors.append(f"Invalid log_level: {self.log_level}")
        
        if not self.checkpoint_dir:
            all_errors.append("checkpoint_dir cannot be empty")
        
        if not self.log_dir:
            all_errors.append("log_dir cannot be empty")
        
        if all_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in all_errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
