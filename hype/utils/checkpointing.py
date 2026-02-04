"""
Checkpointing and model persistence utilities for HyPE system.

This module provides functionality for saving and loading model checkpoints,
principle memory state, and replay buffer data.
"""

import torch
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints and system state."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.checkpoint_dir / "models"
        self.memory_dir = self.checkpoint_dir / "memory"
        self.buffer_dir = self.checkpoint_dir / "buffers"
        
        self.models_dir.mkdir(exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)
        self.buffer_dir.mkdir(exist_ok=True)
    
    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        model_name: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            model_name: Name identifier for the model
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{model_name}_{timestamp}"
        if epoch is not None:
            checkpoint_name += f"_epoch{epoch}"
        checkpoint_name += ".pt"
        
        checkpoint_path = self.models_dir / checkpoint_name
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "timestamp": timestamp,
            "epoch": epoch,
            "metadata": metadata or {}
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved model checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_model_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load model on
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded model checkpoint: {checkpoint_path}")
        
        return {
            "model_name": checkpoint.get("model_name"),
            "timestamp": checkpoint.get("timestamp"),
            "epoch": checkpoint.get("epoch"),
            "metadata": checkpoint.get("metadata", {})
        }
    
    def list_checkpoints(self, model_name: Optional[str] = None) -> List[str]:
        """
        List available checkpoints.
        
        Args:
            model_name: Optional filter by model name
            
        Returns:
            List of checkpoint paths
        """
        if model_name:
            pattern = f"{model_name}_*.pt"
        else:
            pattern = "*.pt"
        
        checkpoints = sorted(self.models_dir.glob(pattern))
        return [str(cp) for cp in checkpoints]
    
    def get_latest_checkpoint(self, model_name: str) -> Optional[str]:
        """
        Get path to latest checkpoint for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_checkpoints(model_name)
        return checkpoints[-1] if checkpoints else None
    
    def delete_old_checkpoints(self, model_name: str, keep_last: int = 5):
        """
        Delete old checkpoints, keeping only the most recent.
        
        Args:
            model_name: Model name
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints(model_name)
        
        if len(checkpoints) > keep_last:
            to_delete = checkpoints[:-keep_last]
            for checkpoint_path in to_delete:
                Path(checkpoint_path).unlink()
                logger.info(f"Deleted old checkpoint: {checkpoint_path}")
    
    def save_checkpoint(
        self,
        system: Any,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save complete system checkpoint including models and principle memory.
        
        Args:
            system: HyPESystem instance
            metadata: Optional metadata dictionary
            checkpoint_name: Optional checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_{timestamp}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Collect system state
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save models if available
        if hasattr(system, 'policy_model') and system.policy_model is not None:
            if hasattr(system.policy_model, 'base_loader') and system.policy_model.base_loader.model is not None:
                checkpoint["policy_model_state"] = system.policy_model.base_loader.model.state_dict()
        
        if hasattr(system, 'value_model') and system.value_model is not None:
            if hasattr(system.value_model, 'base_loader') and system.value_model.base_loader.model is not None:
                checkpoint["value_model_state"] = system.value_model.base_loader.model.state_dict()
        
        # Save principle memory stats (principles are in Milvus Lite database)
        if hasattr(system, 'principle_memory') and system.principle_memory is not None:
            try:
                stats = system.principle_memory.get_stats()
                checkpoint["principle_memory_stats"] = stats
                logger.info(f"Principle memory: {stats['num_principles']} principles")
            except Exception as e:
                logger.warning(f"Could not get principle memory stats: {e}")
        
        # Save trajectories if available
        if hasattr(system, 'trajectories'):
            checkpoint["num_trajectories"] = len(system.trajectories)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved system checkpoint: {checkpoint_path}")
        
        # Also save a summary JSON for easy inspection
        summary_path = checkpoint_path.with_suffix('.json')
        summary = {
            "checkpoint_name": checkpoint_name,
            "timestamp": checkpoint["timestamp"],
            "metadata": checkpoint["metadata"],
            "principle_memory_stats": checkpoint.get("principle_memory_stats", {}),
            "num_trajectories": checkpoint.get("num_trajectories", 0),
            "has_policy_model": "policy_model_state" in checkpoint,
            "has_value_model": "value_model_state" in checkpoint
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved checkpoint summary: {summary_path}")
        
        return str(checkpoint_path)


class PrincipleMemoryPersistence:
    """Handle persistence of Principle Memory state."""
    
    def __init__(self, memory_dir: str = "./checkpoints/memory"):
        """
        Initialize memory persistence handler.
        
        Args:
            memory_dir: Directory for memory snapshots
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def save_principles(
        self,
        principles: List[Any],
        snapshot_name: Optional[str] = None
    ) -> str:
        """
        Save principles to disk.
        
        Args:
            principles: List of Principle objects
            snapshot_name: Optional snapshot name
            
        Returns:
            Path to saved snapshot
        """
        if snapshot_name is None:
            snapshot_name = f"principles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot_path = self.memory_dir / f"{snapshot_name}.pkl"
        
        # Convert principles to serializable format
        serializable_principles = []
        for p in principles:
            principle_dict = {
                "id": p.id,
                "text": p.text,
                "embedding": p.embedding.tolist() if hasattr(p.embedding, 'tolist') else p.embedding,
                "credit_score": p.credit_score,
                "application_count": p.application_count,
                "created_at": p.created_at.isoformat(),
                "last_used": p.last_used.isoformat(),
            }
            if hasattr(p, 'source_trajectory_id'):
                principle_dict["source_trajectory_id"] = p.source_trajectory_id
            serializable_principles.append(principle_dict)
        
        with open(snapshot_path, 'wb') as f:
            pickle.dump(serializable_principles, f)
        
        logger.info(f"Saved {len(principles)} principles to {snapshot_path}")
        return str(snapshot_path)
    
    def load_principles(self, snapshot_path: str) -> List[Dict[str, Any]]:
        """
        Load principles from disk.
        
        Args:
            snapshot_path: Path to snapshot file
            
        Returns:
            List of principle dictionaries
        """
        with open(snapshot_path, 'rb') as f:
            principles = pickle.load(f)
        
        logger.info(f"Loaded {len(principles)} principles from {snapshot_path}")
        return principles
    
    def list_snapshots(self) -> List[str]:
        """List available principle snapshots."""
        snapshots = sorted(self.memory_dir.glob("principles_*.pkl"))
        return [str(s) for s in snapshots]
    
    def get_latest_snapshot(self) -> Optional[str]:
        """Get path to latest principle snapshot."""
        snapshots = self.list_snapshots()
        return snapshots[-1] if snapshots else None


class ReplayBufferPersistence:
    """Handle persistence of replay buffer data."""
    
    def __init__(self, buffer_dir: str = "./checkpoints/buffers"):
        """
        Initialize buffer persistence handler.
        
        Args:
            buffer_dir: Directory for buffer snapshots
        """
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
    
    def save_buffer(
        self,
        trajectories: List[Any],
        buffer_name: Optional[str] = None
    ) -> str:
        """
        Save replay buffer to disk.
        
        Args:
            trajectories: List of Trajectory objects
            buffer_name: Optional buffer name
            
        Returns:
            Path to saved buffer
        """
        if buffer_name is None:
            buffer_name = f"replay_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        buffer_path = self.buffer_dir / f"{buffer_name}.pkl"
        
        # Convert trajectories to serializable format
        serializable_trajectories = []
        for traj in trajectories:
            traj_dict = {
                "id": traj.id,
                "task": traj.task,
                "final_reward": traj.final_reward,
                "success": traj.success,
                "steps": []
            }
            
            for step in traj.steps:
                step_dict = {
                    "state": {
                        "observation": step.state.observation,
                        "history": step.state.history,
                        "timestamp": step.state.timestamp.isoformat()
                    },
                    "action": {
                        "type": step.action.type,
                        "parameters": step.action.parameters,
                        "description": step.action.description
                    },
                    "reward": step.reward,
                    "done": step.done,
                    "hypothesis": step.hypothesis
                }
                traj_dict["steps"].append(step_dict)
            
            # Store principles used
            if hasattr(traj, 'principles_used'):
                traj_dict["principles_used"] = [
                    [p.id for p in step_principles]
                    for step_principles in traj.principles_used
                ]
            
            serializable_trajectories.append(traj_dict)
        
        with open(buffer_path, 'wb') as f:
            pickle.dump(serializable_trajectories, f)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {buffer_path}")
        return str(buffer_path)
    
    def load_buffer(self, buffer_path: str) -> List[Dict[str, Any]]:
        """
        Load replay buffer from disk.
        
        Args:
            buffer_path: Path to buffer file
            
        Returns:
            List of trajectory dictionaries
        """
        with open(buffer_path, 'rb') as f:
            trajectories = pickle.load(f)
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {buffer_path}")
        return trajectories
    
    def list_buffers(self) -> List[str]:
        """List available buffer snapshots."""
        buffers = sorted(self.buffer_dir.glob("replay_buffer_*.pkl"))
        return [str(b) for b in buffers]
    
    def get_latest_buffer(self) -> Optional[str]:
        """Get path to latest buffer snapshot."""
        buffers = self.list_buffers()
        return buffers[-1] if buffers else None


class SystemStateManager:
    """Manage complete system state including models, memory, and buffers."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize system state manager.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
        """
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.memory_persistence = PrincipleMemoryPersistence(
            str(self.checkpoint_manager.memory_dir)
        )
        self.buffer_persistence = ReplayBufferPersistence(
            str(self.checkpoint_manager.buffer_dir)
        )
    
    def save_full_state(
        self,
        models: Dict[str, torch.nn.Module],
        principles: List[Any],
        trajectories: List[Any],
        state_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save complete system state.
        
        Args:
            models: Dictionary of models to save
            principles: List of principles
            trajectories: List of trajectories
            state_name: Optional state name
            metadata: Optional metadata
            
        Returns:
            Dictionary with paths to saved components
        """
        if state_name is None:
            state_name = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        saved_paths = {}
        
        # Save models
        for model_name, model in models.items():
            checkpoint_path = self.checkpoint_manager.save_model_checkpoint(
                model=model,
                model_name=f"{state_name}_{model_name}",
                metadata=metadata
            )
            saved_paths[f"model_{model_name}"] = checkpoint_path
        
        # Save principles
        principles_path = self.memory_persistence.save_principles(
            principles=principles,
            snapshot_name=f"{state_name}_principles"
        )
        saved_paths["principles"] = principles_path
        
        # Save trajectories
        buffer_path = self.buffer_persistence.save_buffer(
            trajectories=trajectories,
            buffer_name=f"{state_name}_buffer"
        )
        saved_paths["buffer"] = buffer_path
        
        # Save state manifest
        manifest = {
            "state_name": state_name,
            "timestamp": datetime.now().isoformat(),
            "components": saved_paths,
            "metadata": metadata or {}
        }
        
        manifest_path = self.checkpoint_manager.checkpoint_dir / f"{state_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        saved_paths["manifest"] = str(manifest_path)
        
        logger.info(f"Saved full system state: {state_name}")
        return saved_paths
    
    def load_full_state(
        self,
        state_name: str,
        models: Dict[str, torch.nn.Module],
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load complete system state.
        
        Args:
            state_name: Name of state to load
            models: Dictionary of models to load into
            device: Device to load models on
            
        Returns:
            Dictionary with loaded components
        """
        # Load manifest
        manifest_path = self.checkpoint_manager.checkpoint_dir / f"{state_name}_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        loaded_state = {
            "metadata": manifest.get("metadata", {}),
            "timestamp": manifest.get("timestamp")
        }
        
        # Load models
        for model_name, model in models.items():
            checkpoint_key = f"model_{model_name}"
            if checkpoint_key in manifest["components"]:
                checkpoint_path = manifest["components"][checkpoint_key]
                self.checkpoint_manager.load_model_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    device=device
                )
        
        # Load principles
        if "principles" in manifest["components"]:
            principles_path = manifest["components"]["principles"]
            loaded_state["principles"] = self.memory_persistence.load_principles(
                principles_path
            )
        
        # Load trajectories
        if "buffer" in manifest["components"]:
            buffer_path = manifest["components"]["buffer"]
            loaded_state["trajectories"] = self.buffer_persistence.load_buffer(
                buffer_path
            )
        
        logger.info(f"Loaded full system state: {state_name}")
        return loaded_state
    
    def list_states(self) -> List[str]:
        """List available system states."""
        manifests = sorted(self.checkpoint_manager.checkpoint_dir.glob("state_*_manifest.json"))
        return [m.stem.replace("_manifest", "") for m in manifests]
    
    def get_latest_state(self) -> Optional[str]:
        """Get name of latest system state."""
        states = self.list_states()
        return states[-1] if states else None
