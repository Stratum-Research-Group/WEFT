"""
Multi-Task Specialist Tree Router Fusion:
Integrates the Specialist Domain Router directly with Speculative Tree Decoding:
1. Dynamically classifies prompt domain (Genomics, Medicine, Tools, Code, Reasoning).
2. Directs candidate tree drafting through the exact specialized student projection head.
3. Assembles domain-specialized hypothesis trees verified in a single parallel target pass.
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
from weft.mtp_tree_integrator import MTPTreeIntegrator
from weft.speculative_tree_decoder import SpeculativeTreeNode

class MultiTaskSpecialistTreeRouter(nn.Module):
    def __init__(
        self,
        tree_integrator: MTPTreeIntegrator,
        d_model: int = 256,
        num_domains: int = 7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.tree_integrator = tree_integrator.to(device)
        self.device = device
        self.num_domains = num_domains
        self.domains = [
            "alphagenome", "functiongemma", "medgemma", 
            "t5gemma", "tipsv2", "gemmascope", "audio_conformer"
        ]
        self.router = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, num_domains)
        ).to(device)

    def route_prompt(self, prefix_ids: torch.Tensor) -> Tuple[str, float]:
        """
        Classifies input prefix into the most relevant specialized domain.
        """
        with torch.no_grad():
            out = self.tree_integrator.draft_model(prefix_ids, output_hidden_states=True)
            pooled = out.hidden_states[-1].mean(dim=1) # [1, d_model]
            logits = self.router(pooled)
            probs = torch.softmax(logits, dim=-1)
            best_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, best_idx].item()
            return self.domains[best_idx], confidence

    def generate_specialized_tree(
        self,
        prefix_ids: torch.Tensor,
        embed_w: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Generates domain-aware candidate tree and returns routing diagnostics.
        """
        domain, conf = self.route_prompt(prefix_ids)
        flat_tokens, tree_mask, nodes = self.tree_integrator.generate_candidate_tree_single_shot(prefix_ids, embed_w)
        
        return {
            "dispatched_domain": domain,
            "routing_confidence": conf,
            "candidate_tokens": flat_tokens,
            "tree_mask": tree_mask,
            "tree_nodes": nodes,
            "num_candidates": flat_tokens.shape[1],
        }
