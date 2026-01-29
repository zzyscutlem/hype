"""
H-MCTS (Hypothesis-driven Monte Carlo Tree Search) planner.

This module implements the hierarchical planning component of the HyPE system.
H-MCTS performs tree search over hypotheses (high-level plans) before
instantiating specific actions.
"""

import math
import logging
from typing import List, Optional, Tuple
from datetime import datetime

from ..core.data_models import HypothesisNode, State, Principle, Action
from ..models.policy_model import PolicyModel
from ..models.value_model import ValueModel


logger = logging.getLogger(__name__)


class TreeUtilities:
    """
    Utilities for H-MCTS tree operations.
    
    Provides methods for tree traversal, node manipulation, and analysis.
    """
    
    @staticmethod
    def get_path_to_root(node: HypothesisNode) -> List[HypothesisNode]:
        """
        Get the path from a node to the root.
        
        Args:
            node: Starting node
            
        Returns:
            List of nodes from the given node to root (inclusive)
        """
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    @staticmethod
    def get_depth(node: HypothesisNode) -> int:
        """
        Get the depth of a node in the tree.
        
        Args:
            node: Node to measure depth
            
        Returns:
            Depth (root has depth 0)
        """
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    @staticmethod
    def get_all_descendants(node: HypothesisNode) -> List[HypothesisNode]:
        """
        Get all descendants of a node (BFS traversal).
        
        Args:
            node: Root node for traversal
            
        Returns:
            List of all descendant nodes
        """
        descendants = []
        queue = [node]
        
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            queue.extend(current.children)
        
        return descendants
    
    @staticmethod
    def get_leaf_nodes(root: HypothesisNode) -> List[HypothesisNode]:
        """
        Get all leaf nodes in the tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            List of leaf nodes
        """
        leaves = []
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            if current.is_leaf():
                leaves.append(current)
            else:
                queue.extend(current.children)
        
        return leaves
    
    @staticmethod
    def get_tree_size(root: HypothesisNode) -> int:
        """
        Get the total number of nodes in the tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Total number of nodes
        """
        return len(TreeUtilities.get_all_descendants(root))
    
    @staticmethod
    def get_max_depth(root: HypothesisNode) -> int:
        """
        Get the maximum depth of the tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Maximum depth
        """
        max_depth = 0
        queue = [(root, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child in current.children:
                queue.append((child, depth + 1))
        
        return max_depth
    
    @staticmethod
    def find_best_child(node: HypothesisNode, by_visits: bool = True) -> Optional[HypothesisNode]:
        """
        Find the best child of a node.
        
        Args:
            node: Parent node
            by_visits: If True, select by visit count; if False, by Q-value
            
        Returns:
            Best child node, or None if no children
        """
        if not node.children:
            return None
        
        if by_visits:
            return max(node.children, key=lambda c: c.visit_count)
        else:
            return max(node.children, key=lambda c: c.q_value)
    
    @staticmethod
    def print_tree(root: HypothesisNode, max_depth: int = 3, indent: int = 0) -> None:
        """
        Print tree structure for debugging.
        
        Args:
            root: Root node to print from
            max_depth: Maximum depth to print
            indent: Current indentation level
        """
        if indent > max_depth:
            return
        
        prefix = "  " * indent
        print(f"{prefix}[V={root.visit_count}, Q={root.q_value:.3f}] {root.hypothesis[:50]}...")
        
        for child in root.children:
            TreeUtilities.print_tree(child, max_depth, indent + 1)


class HMCTS:
    """
    Hypothesis-driven Monte Carlo Tree Search planner.
    
    Performs hierarchical planning by:
    1. Generating hypotheses (high-level plans)
    2. Evaluating hypotheses using Value Model
    3. Selecting promising hypotheses using UCB
    4. Building a search tree to find the best hypothesis
    """
    
    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        exploration_constant: float = 1.414,  # sqrt(2)
        max_depth: int = 5,
        num_hypotheses_per_expansion: int = 3,
        early_stop_threshold: float = 0.8,  # Stop if best Q-value exceeds this
        min_iterations: int = 10,  # Minimum iterations before early stopping
        value_cache_size: int = 100  # Cache size for value predictions
    ):
        """
        Initialize H-MCTS planner.
        
        Args:
            policy_model: Policy model for hypothesis generation
            value_model: Value model for hypothesis evaluation
            exploration_constant: UCB exploration constant (c)
            max_depth: Maximum tree depth
            num_hypotheses_per_expansion: Number of hypotheses to generate per expansion
            early_stop_threshold: Q-value threshold for early stopping
            min_iterations: Minimum iterations before allowing early stop
            value_cache_size: Maximum size of value prediction cache
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.num_hypotheses_per_expansion = num_hypotheses_per_expansion
        self.early_stop_threshold = early_stop_threshold
        self.min_iterations = min_iterations
        self.tree_utils = TreeUtilities()
        
        # Value prediction cache to avoid redundant model calls
        self.value_cache = {}
        self.value_cache_size = value_cache_size
        
        logger.info(
            f"Initialized HMCTS with c={exploration_constant}, "
            f"max_depth={max_depth}, num_hypotheses={num_hypotheses_per_expansion}, "
            f"early_stop={early_stop_threshold}, min_iter={min_iterations}"
        )
    
    def search(
        self,
        task: str,
        state: State,
        principles: List[Principle],
        budget: int
    ) -> HypothesisNode:
        """
        Perform MCTS search to find the best hypothesis.
        
        This is the main entry point for H-MCTS planning. It runs the
        MCTS loop (select, expand, simulate, backpropagate) for the
        specified budget and returns the best hypothesis.
        
        Includes early stopping if a high-quality hypothesis is found.
        
        Args:
            task: Task description
            state: Current environment state
            principles: Retrieved principles for this task
            budget: Number of search iterations
            
        Returns:
            Best hypothesis node based on visit count
        """
        # Clear value cache for new search
        self.value_cache.clear()
        
        # Create root node with initial hypothesis
        root = HypothesisNode(
            hypothesis="Initial state - planning next action",
            state=state,
            parent=None,
            children=[],
            visit_count=0,
            total_value=0.0,
            principles=principles,
            ucb_score=0.0
        )
        
        logger.info(f"Starting H-MCTS search with budget={budget}")
        print(f"      ⏳ H-MCTS search (budget={budget})...", flush=True)
        
        best_q_value = float('-inf')
        iterations_without_improvement = 0
        
        # Run MCTS iterations
        for iteration in range(budget):
            # Selection: traverse tree to find leaf node
            node = self.select(root)
            
            # Expansion: generate new hypotheses if not at max depth
            if not self._is_terminal(node) and TreeUtilities.get_depth(node) < self.max_depth:
                node = self.expand(node, task, principles)
            
            # Simulation: evaluate the node
            value = self.simulate(node, task, principles)
            
            # Backpropagation: update values up the tree
            self.backpropagate(node, value)
            
            # Check for early stopping after minimum iterations
            if iteration >= self.min_iterations:
                best_child = TreeUtilities.find_best_child(root, by_visits=True)
                if best_child:
                    current_best_q = best_child.q_value
                    
                    # Update best Q-value
                    if current_best_q > best_q_value:
                        best_q_value = current_best_q
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                    
                    # Early stop if:
                    # 1. Q-value exceeds threshold, OR
                    # 2. No improvement for 20 iterations
                    if (current_best_q >= self.early_stop_threshold or 
                        iterations_without_improvement >= 20):
                        logger.info(
                            f"Early stopping at iteration {iteration + 1}/{budget} "
                            f"(Q={current_best_q:.3f}, no_improve={iterations_without_improvement})"
                        )
                        print(f"      ✅ Early stop at {iteration + 1}/{budget} (Q={current_best_q:.3f})", flush=True)
                        break
            
            if (iteration + 1) % 10 == 0:
                logger.debug(f"Iteration {iteration + 1}/{budget} complete")
                print(f"      ⏳ H-MCTS iteration {iteration + 1}/{budget}...", flush=True)
        
        # Select best child based on visit count
        best_child = TreeUtilities.find_best_child(root, by_visits=True)
        
        if best_child is None:
            logger.warning("No children generated, returning root")
            print(f"      ⚠️  No children generated, using root", flush=True)
            return root
        
        logger.info(
            f"Search complete. Best hypothesis: '{best_child.hypothesis[:50]}...' "
            f"(visits={best_child.visit_count}, Q={best_child.q_value:.3f})"
        )
        print(f"      ✅ H-MCTS complete (visits={best_child.visit_count}, Q={best_child.q_value:.3f})", flush=True)
        
        return best_child
    
    def select(self, node: HypothesisNode) -> HypothesisNode:
        """
        Select a leaf node using UCB selection.
        
        Traverses the tree from the given node, selecting children with
        the highest UCB score until a leaf node is reached.
        
        Args:
            node: Starting node (typically root)
            
        Returns:
            Selected leaf node
        """
        current = node
        
        while not current.is_leaf():
            # Select child with highest UCB score
            current = self._select_best_ucb_child(current)
        
        return current
    
    def _select_best_ucb_child(self, node: HypothesisNode) -> HypothesisNode:
        """
        Select the child with the highest UCB score.
        
        UCB formula: Q(node) + c * sqrt(ln(N(parent)) / N(node))
        
        Args:
            node: Parent node
            
        Returns:
            Child with highest UCB score
        """
        if not node.children:
            raise ValueError("Cannot select from node with no children")
        
        best_child = None
        best_ucb = float('-inf')
        
        for child in node.children:
            ucb = self._compute_ucb(child, node)
            child.ucb_score = ucb
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child
    
    def _compute_ucb(self, node: HypothesisNode, parent: HypothesisNode) -> float:
        """
        Compute UCB score for a node.
        
        Formula: Q(node) + c * sqrt(ln(N(parent)) / N(node))
        
        Args:
            node: Node to compute UCB for
            parent: Parent node
            
        Returns:
            UCB score
        """
        if node.visit_count == 0:
            # Unvisited nodes get infinite UCB to ensure exploration
            return float('inf')
        
        # Exploitation term: average value
        q_value = node.q_value
        
        # Exploration term: UCB bonus
        exploration_bonus = self.exploration_constant * math.sqrt(
            math.log(parent.visit_count) / node.visit_count
        )
        
        return q_value + exploration_bonus
    
    def expand(
        self,
        node: HypothesisNode,
        task: str,
        principles: List[Principle]
    ) -> HypothesisNode:
        """
        Expand a node by generating hypothesis children.
        
        Uses the Policy Model to generate multiple hypotheses as children
        of the given node.
        
        Args:
            node: Node to expand
            task: Task description
            principles: Retrieved principles
            
        Returns:
            One of the newly created children (for immediate simulation)
        """
        # Format state as string for model input
        state_str = self._format_state(node.state)
        
        # Generate multiple hypotheses
        hypotheses = self.policy_model.generate_hypothesis(
            task=task,
            state=state_str,
            principles=principles,
            num_hypotheses=self.num_hypotheses_per_expansion,
            temperature=0.8  # Higher temperature for diversity
        )
        
        # Create child nodes for each hypothesis
        for hypothesis in hypotheses:
            child = HypothesisNode(
                hypothesis=hypothesis,
                state=node.state,  # Same state as parent
                parent=node,
                children=[],
                visit_count=0,
                total_value=0.0,
                principles=principles,
                ucb_score=0.0
            )
            node.children.append(child)
        
        logger.debug(f"Expanded node with {len(hypotheses)} children")
        
        # Return first child for immediate simulation
        return node.children[0] if node.children else node
    
    def simulate(
        self,
        node: HypothesisNode,
        task: str,
        principles: List[Principle]
    ) -> float:
        """
        Simulate (evaluate) a hypothesis using the Value Model.
        
        Uses caching to avoid redundant model calls for the same hypothesis.
        
        Args:
            node: Node to evaluate
            task: Task description
            principles: Retrieved principles
            
        Returns:
            Predicted value for this hypothesis
        """
        # Create cache key from hypothesis and state
        cache_key = (node.hypothesis, str(node.state.observation))
        
        # Check cache first
        if cache_key in self.value_cache:
            logger.debug("Value cache hit")
            return self.value_cache[cache_key]
        
        # Format state as string
        state_str = self._format_state(node.state)
        
        # Predict value using Value Model
        value = self.value_model.predict_value(
            task=task,
            state=state_str,
            hypothesis=node.hypothesis,
            principles=principles
        )
        
        # Cache the result
        if len(self.value_cache) < self.value_cache_size:
            self.value_cache[cache_key] = value
        
        return value
    
    def backpropagate(self, node: HypothesisNode, value: float) -> None:
        """
        Backpropagate value up the tree.
        
        Updates visit counts and total values for the node and all its
        ancestors up to the root.
        
        Args:
            node: Starting node
            value: Value to backpropagate
        """
        current = node
        
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent
    
    def _format_state(self, state: State) -> str:
        """
        Format state as string for model input.
        
        Args:
            state: State object
            
        Returns:
            String representation of state
        """
        # Simple formatting - can be enhanced based on environment
        obs_str = str(state.observation)
        history_str = ", ".join(state.history[-3:]) if state.history else "No previous actions"
        
        return f"Observation: {obs_str}\nRecent actions: {history_str}"
    
    def _is_terminal(self, node: HypothesisNode) -> bool:
        """
        Check if a node represents a terminal state.
        
        Args:
            node: Node to check
            
        Returns:
            True if terminal, False otherwise
        """
        # For now, we don't have terminal detection in hypotheses
        # This would be environment-specific
        return False
    
    def get_best_hypothesis(self, root: HypothesisNode) -> HypothesisNode:
        """
        Get the best hypothesis from a search tree.
        
        Args:
            root: Root node of the search tree
            
        Returns:
            Best child based on visit count
        """
        return TreeUtilities.find_best_child(root, by_visits=True)
    
    def get_tree_statistics(self, root: HypothesisNode) -> dict:
        """
        Get statistics about the search tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Dictionary of statistics
        """
        return {
            "total_nodes": TreeUtilities.get_tree_size(root),
            "max_depth": TreeUtilities.get_max_depth(root),
            "num_leaves": len(TreeUtilities.get_leaf_nodes(root)),
            "root_visits": root.visit_count,
            "root_value": root.q_value
        }
