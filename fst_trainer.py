import os
import torch
import copy
from typing import List, Tuple, Callable
import tiktoken

class GEPAMutator:
    """
    Generalized Eligibility Prompt Algorithm (GEPA) Mutator.
    Optimizes the 'fast weights' (system prompt) using an evolutionary search.
    """
    
    def __init__(
        self,
        base_prompt: str,
        api_key: str = None,
        use_external_api: bool = False
    ):
        self.current_best_prompt = base_prompt
        self.api_key = api_key
        self.use_external_api = use_external_api
        
    def _mutate_local(self, model, tok, device, prompt: str, num_mutations: int) -> List[str]:
        """Use the local model to suggest mutated prompts."""
        mutation_prompt = (
            "You are an AI optimization assistant. Below is a system prompt used to guide an AI's reasoning. "
            "Your task is to rephrase or improve it to make it more effective, without changing the core XML tag requirements.\n\n"
            f"Original Prompt:\n{prompt}\n\nImproved Prompt:\n"
        )
        
        mutations = []
        tokens = tok.encode(mutation_prompt, allowed_special={"<|endoftext|>"})
        
        # This uses simple rollout generation to get new prompts
        for _ in range(num_mutations):
            from rl_trainer import rollout
            raw = model.module if hasattr(model, "module") else model
            mutated, _ = rollout(raw, tokens, tok, max_new_tokens=100, temperature=1.0, device=device)
            # Minimal cleaning
            clean_mutated = mutated.split("Original Prompt:")[0].strip()
            if clean_mutated:
                mutations.append(clean_mutated + "\n\n")
            else:
                mutations.append(prompt)
                
        return mutations
        
    def _mutate_external(self, prompt: str, num_mutations: int) -> List[str]:
        """Stub for external API call (e.g. OpenAI) to get mutated prompts."""
        # Stub implementation
        print("  [FST] Calling external API for high-quality prompt mutation (STUB)...")
        mutations = []
        for i in range(num_mutations):
            mutations.append(prompt.strip() + f"\n(Mutation variant {i+1})\n\n")
        return mutations
        
    def propose_mutations(self, model, tok, device, num_mutations: int = 3) -> List[str]:
        """Generate candidates for the next generation prompt."""
        if self.use_external_api:
            return self._mutate_external(self.current_best_prompt, num_mutations)
        else:
            return self._mutate_local(model, tok, device, self.current_best_prompt, num_mutations)

    def evaluate_and_update(
        self, 
        candidates: List[str], 
        eval_function: Callable[[str], float]
    ) -> str:
        """
        Evaluate all candidates using the provided evaluation function (which runs rollouts).
        Updates and returns the best prompt.
        """
        best_score = eval_function(self.current_best_prompt)
        best_prompt = self.current_best_prompt
        
        for cand in candidates:
            score = eval_function(cand)
            if score > best_score:
                best_score = score
                best_prompt = cand
                
        self.current_best_prompt = best_prompt
        return best_prompt
