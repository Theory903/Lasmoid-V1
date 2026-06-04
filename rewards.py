"""
rewards.py — LasmoidV1 Reasoning RL Reward Functions
=====================================================
All reward signals used in GRPO / RLOO training.

Reward taxonomy:
  format_reward   : Does the response use <think>…</think> format?
  length_reward   : Is the thinking trace non-trivially long?
  accuracy_reward : Does the final answer match the ground truth?
  total_reward    : Weighted combination of all three.

Design principles:
  • All rewards in [-2, +2] range — normalise before computing advantages
  • format + length rewards fire even without ground truth (unsupervised)
  • accuracy reward is optional (only when gt_answer is known)
  • All functions are vectorised over a list of responses
"""

import re
import math
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# TOKEN / FORMAT CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# The thinking format Parallel-R1 is trained to use:
#   <Parallel>
#   <Path>
#   ... reasoning trace (any length) ...
#   </Path>
#   ... multiple paths ...
#   </Parallel>
#   <Summary>
#   ... summary ...
#   </Summary>
#   Final answer: X
PARALLEL_OPEN  = "<Parallel>"
PARALLEL_CLOSE = "</Parallel>"
PATH_OPEN      = "<Path>"
PATH_CLOSE     = "</Path>"
SUMMARY_OPEN   = "<Summary>"
SUMMARY_CLOSE  = "</Summary>"

# Minimum token-equivalent characters for a "non-trivial" trace
# (roughly 10 tokens × 4 chars/token)
MIN_TRACE_CHARS = 40

# Length reward saturates at this many characters (~200 tokens)
MAX_TRACE_CHARS = 800


# ══════════════════════════════════════════════════════════════════════
# FORMAT REWARD
# ══════════════════════════════════════════════════════════════════════

def format_reward(response: str) -> float:
    """
    Reward for using the Parallel-R1 format.

    Scoring:
      +1.0  all tags present, multiple paths, non-empty summary, answer after </Summary>
      +0.8  all tags present, single path
      +0.5  <Parallel> tag present but format incomplete
       0.0  no thinking structure at all
      -0.5  tags present but traces are empty

    Returns float in [-0.5, +1.0].
    """
    has_para_open  = PARALLEL_OPEN in response
    has_para_close = PARALLEL_CLOSE in response

    if not has_para_open:
        return 0.0

    if has_para_open and not has_para_close:
        return 0.5  # partial credit: model started reasoning

    # Both tags present — check trace content
    try:
        paths = _extract_reasoning_paths(response)
    except ValueError:
        return 0.0

    if not paths:
        return -0.5

    for p in paths:
        if len(p.strip()) < 5:
            return -0.5

    # Check for summary
    has_sum_open = SUMMARY_OPEN in response
    has_sum_close = SUMMARY_CLOSE in response

    if not (has_sum_open and has_sum_close):
        return 0.6  # has paths but no summary

    try:
        summary = _extract_summary(response)
    except ValueError:
        return 0.0

    if len(summary.strip()) < 5:
        return -0.5

    # Check that there's actually content after </Summary>
    after_close = response.split(SUMMARY_CLOSE, 1)[-1].strip()
    if not after_close:
        return 0.7  # has trace/summary but no final answer — partial

    if len(paths) > 1:
        return 1.0
    else:
        return 0.8


# ══════════════════════════════════════════════════════════════════════
# LENGTH REWARD
# ══════════════════════════════════════════════════════════════════════

def length_reward(response: str, min_chars: int = MIN_TRACE_CHARS,
                  max_chars: int = MAX_TRACE_CHARS) -> float:
    """
    Reward for a thinking trace of appropriate length.

    Prevents two failure modes:
      1. Empty/trivial traces
      2. Excessively long traces (reward hacking via repetition)

    Returns a float in [0.0, +1.0]:
      0.0  — no parallel tags, or total trace < min_chars
      linear ramp from 0→1 as trace length goes min_chars → max_chars
      1.0  — trace ≥ max_chars (capped, no bonus for being longer)
    """
    if PARALLEL_OPEN not in response or PARALLEL_CLOSE not in response:
        return 0.0
    try:
        paths = _extract_reasoning_paths(response)
        summary = _extract_summary(response) if (SUMMARY_OPEN in response and SUMMARY_CLOSE in response) else ""
    except ValueError:
        return 0.0

    total_chars = sum(len(p.strip()) for p in paths) + len(summary.strip())
    
    if total_chars < min_chars:
        return 0.0
    return min(1.0, (total_chars - min_chars) / max(1, max_chars - min_chars))


# ══════════════════════════════════════════════════════════════════════
# ACCURACY REWARD
# ══════════════════════════════════════════════════════════════════════

def accuracy_reward(response: str, gt_answer: Optional[str]) -> float:
    """
    Reward for a correct final answer.

    Matching strategy (in order):
      1. Exact string match (after normalisation)
      2. Numeric match (float comparison with 1e-3 tolerance)
      3. Last-number match (extract last number in response)

    Returns:
      +2.0  correct answer
       0.0  wrong answer or no ground truth
    """
    if gt_answer is None:
        return 0.0

    gt_norm = _normalise_answer(gt_answer)

    # Strategy 1: check text after </Summary> first
    answer_section = response
    if SUMMARY_CLOSE in response:
        answer_section = response.split(SUMMARY_CLOSE, 1)[-1]

    resp_norm = _normalise_answer(answer_section)

    # Exact match
    if gt_norm in resp_norm or resp_norm in gt_norm:
        return 2.0

    # Numeric match
    gt_num  = _extract_number(gt_norm)
    resp_num = _extract_number(resp_norm)
    if gt_num is not None and resp_num is not None:
        if math.isclose(gt_num, resp_num, rel_tol=1e-3, abs_tol=1e-6):
            return 2.0

    # Last-number fallback (common in math tasks)
    last_num = _last_number(response)
    if last_num is not None and gt_num is not None:
        if math.isclose(last_num, gt_num, rel_tol=1e-3, abs_tol=1e-6):
            return 2.0

    return 0.0


# ══════════════════════════════════════════════════════════════════════
# TOTAL REWARD (weighted combination)
# ══════════════════════════════════════════════════════════════════════

def compute_reward(
    response: str,
    gt_answer: Optional[str] = None,
    w_format: float = 1.0,
    w_length: float = 0.5,
    w_accuracy: float = 2.0,
) -> Tuple[float, dict]:
    """
    Compute the total reward for a single model response.

    Args:
        response:    Full decoded text from the model.
        gt_answer:   Ground truth answer string (None = format/length only).
        w_format:    Weight for format reward.
        w_length:    Weight for length reward.
        w_accuracy:  Weight for accuracy reward (usually 2× format).

    Returns:
        (total_reward, breakdown_dict)
    """
    r_fmt = format_reward(response)
    r_len = length_reward(response)
    r_acc = accuracy_reward(response, gt_answer)

    total = w_format * r_fmt + w_length * r_len + w_accuracy * r_acc
    breakdown = {
        "format":   r_fmt,
        "length":   r_len,
        "accuracy": r_acc,
        "total":    total,
    }
    return total, breakdown


def batch_rewards(
    responses: List[str],
    gt_answers: Optional[List[Optional[str]]] = None,
    **kwargs,
) -> Tuple[List[float], List[dict]]:
    """
    Compute rewards for a batch of responses.

    Args:
        responses:   List of G decoded strings per prompt.
        gt_answers:  Ground truth list (same length as responses), or None.

    Returns:
        (rewards_list, breakdown_list)
    """
    if gt_answers is None:
        gt_answers = [None] * len(responses)
    rewards, breakdowns = [], []
    for resp, gt in zip(responses, gt_answers):
        r, b = compute_reward(resp, gt, **kwargs)
        rewards.append(r)
        breakdowns.append(b)
    return rewards, breakdowns


# ══════════════════════════════════════════════════════════════════════
# GRPO ADVANTAGE COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_grpo_advantages(
    rewards: List[float],
    eps: float = 1e-8,
) -> List[float]:
    """
    Compute group-relative advantages for GRPO.

    Given G rewards for G completions of the same prompt:
      advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

    This normalises within the group so the model learns *which* completion
    is better, not the absolute reward magnitude.

    For a single sample (G=1), returns [0.0] — no signal.

    Returns list of float advantages, same length as rewards.
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)

    import statistics
    mean_r = statistics.mean(rewards)
    std_r  = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    return [(r - mean_r) / (std_r + eps) for r in rewards]


def compute_rloo_advantages(
    rewards: List[float],
    eps: float = 1e-8,
) -> List[float]:
    """
    REINFORCE Leave-One-Out (RLOO) advantage estimator.

    advantage_i = r_i - (sum(r_j for j≠i) / (G-1))

    Lower variance than GRPO at the same group size G.
    Equivalent to GRPO for large G, but more accurate for small G (2-4).

    Returns list of float advantages, same length as rewards.
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    G = len(rewards)
    total = sum(rewards)
    return [(r - (total - r) / (G - 1)) / (max(abs(r - (total - r) / (G - 1)) for r in rewards) + eps)
            for r in rewards]


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _extract_reasoning_paths(response: str) -> List[str]:
    """Extract all content inside <Path>...</Path> tags within <Parallel>...</Parallel>."""
    start = response.find(PARALLEL_OPEN)
    end   = response.find(PARALLEL_CLOSE)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Malformed parallel tags")
        
    parallel_content = response[start + len(PARALLEL_OPEN): end]
    
    paths = []
    # Find all <Path>...</Path> blocks
    p_start = parallel_content.find(PATH_OPEN)
    while p_start != -1:
        p_end = parallel_content.find(PATH_CLOSE, p_start)
        if p_end == -1:
            break
        paths.append(parallel_content[p_start + len(PATH_OPEN): p_end])
        p_start = parallel_content.find(PATH_OPEN, p_end + len(PATH_CLOSE))
        
    return paths

def _extract_summary(response: str) -> str:
    """Extract content inside <Summary>...</Summary> tags."""
    start = response.find(SUMMARY_OPEN)
    end   = response.find(SUMMARY_CLOSE)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Malformed summary tags")
    return response[start + len(SUMMARY_OPEN): end]


def _normalise_answer(text: str) -> str:
    """Lowercase, strip whitespace, remove common answer prefixes."""
    text = text.lower().strip()
    # Remove common answer prefixes
    for prefix in ("the answer is", "answer:", "final answer:", "=", "≈"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")

def _extract_number(text: str) -> Optional[float]:
    """Extract first number from text. Returns None if no number found."""
    m = _NUMBER_RE.search(text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _last_number(text: str) -> Optional[float]:
    """Extract last number from text (common pattern in math answers)."""
    matches = _NUMBER_RE.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# UNIT TESTS (run with: python rewards.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  rewards.py — unit tests")
    print("=" * 55)

    # Format reward tests
    assert format_reward("no structure here") == 0.0
    valid_resp = "<Parallel>\n<Path>path 1 is good</Path>\n<Path>path 2 is also good</Path>\n</Parallel>\n<Summary>overall</Summary>\nFinal answer: 42"
    assert format_reward(valid_resp) == 1.0
    single_path_resp = "<Parallel>\n<Path>single path</Path>\n</Parallel>\n<Summary>overall</Summary>\nFinal answer: 42"
    assert format_reward(single_path_resp) == 0.8
    assert format_reward("<Parallel></Parallel>") == -0.5
    assert format_reward("<Parallel>started but didn't close") == 0.5
    print("format_reward ✓")

    # Length reward tests
    assert length_reward("no tags") == 0.0
    short = "<Parallel><Path>short</Path></Parallel>"
    assert length_reward(short) == 0.0  # below MIN_TRACE_CHARS
    long_trace = "<Parallel><Path>" + "a" * 1000 + "</Path></Parallel>"
    assert length_reward(long_trace) == 1.0  # above MAX_TRACE_CHARS
    print("length_reward ✓")

    # Accuracy reward tests
    resp = "<Parallel><Path>2 + 2 is 4</Path></Parallel><Summary>it is 4</Summary>\nFinal answer: 4"
    assert accuracy_reward(resp, "4") == 2.0
    assert accuracy_reward(resp, "5") == 0.0
    assert accuracy_reward(resp, None) == 0.0
    print("accuracy_reward ✓")

    # GRPO advantages
    advs = compute_grpo_advantages([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(advs)) < 1e-6, "advantages should sum to ~0"
    print("grpo_advantages ✓")

    # Total reward
    r, b = compute_reward(
        "<Parallel><Path>Step 1: solve.</Path></Parallel><Summary>Step 2: answer is 42</Summary>\nFinal answer: 42",
        gt_answer="42",
    )
    assert r > 3.0, f"Expected high total reward, got {r}"
    print(f"total_reward ✓  (r={r:.3f}, breakdown={b})")

    print("\nALL TESTS PASSED ✓")
