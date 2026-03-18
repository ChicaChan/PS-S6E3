# %% cell 6
COMPETITION = "playground-series-s6e3"            # New competition slug
TOP_N = 20                                       # Fewer notebooks available early on
BEST = "high"                                     # Still AUC ROC, higher is better
STRATEGY = "greedy"                               # Still appropriate
MAX_SUBS = 5                                      # Keep as-is
ENSEMBLE_METHOD = "rank"                          # Rank averaging remains ideal for AUC
OPTUNA_TRIALS = 100
MIN_CORR = 0.990                                  # Loosen from 0.9995 — early submissions will be more diverse
EXCLUDE_BLENDS = False
LLM_FILTER = True                                 # Still important
LLM_MODEL = "anthropic/claude-sonnet-4.6"
SHOW_CORR = True
REPORT = True

# %% cell 7
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Install optuna if needed (for rankweight_optuna)
try:
    import optuna
except ImportError:
    import subprocess as _sp
    _sp.check_call(["pip", "install", "-q", "optuna"])
    import optuna

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Load secrets from Kaggle Secrets and configure authentication
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()

    # Kaggle CLI authentication (required for kernels list/output/pull)
    os.environ["KAGGLE_USERNAME"] = _secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = _secrets.get_secret("KAGGLE_KEY")
    print("Loaded Kaggle CLI credentials from Kaggle Secrets")

    # OpenRouter API key (only needed for LLM filter)
    if LLM_FILTER:
        try:
            os.environ["OPENROUTER_API_KEY"] = _secrets.get_secret("OPENROUTER_API_KEY")
            print("Loaded OPENROUTER_API_KEY from Kaggle Secrets")
        except Exception as e:
            print(f"Warning: Could not load OPENROUTER_API_KEY: {e}")
            print("LLM filter will be disabled")
            LLM_FILTER = False

except Exception as e:
    raise RuntimeError(
        f"Could not load Kaggle Secrets: {e}\n"
        "Add these secrets in your Kaggle notebook (Add-ons > Secrets):\n"
        "  - KAGGLE_USERNAME: your Kaggle username\n"
        "  - KAGGLE_KEY: your Kaggle API key (from kaggle.json)\n"
        "  - OPENROUTER_API_KEY: (optional, only if LLM_FILTER=True)"
    )

import requests

WORK_DIR = Path("/kaggle/working")
SUBS_DIR = WORK_DIR / "subs_tmp"
OUTPUT_PATH = WORK_DIR / "submission.csv"

# %% cell 8

def get_kernels_list(competition: str, page_size: int = 50, best: str = "high") -> list:
    """Get list of kernels using kaggle CLI, sorted by score."""
    sort_by = "scoreDescending" if best == "high" else "scoreAscending"
    cmd = ["kaggle", "kernels", "list", "--competition", competition,
           "--page-size", str(page_size), "--sort-by", sort_by, "-v"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error listing kernels: {result.stderr}")
        return []

    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        return []

    headers = lines[0].split(',')
    kernels = []
    for line in lines[1:]:
        values = line.split(',')
        if len(values) >= len(headers):
            kernel = dict(zip(headers, values))
            kernels.append(kernel)
    return kernels


def get_notebook_source(kernel_ref: str) -> str:
    """Download and read notebook source code using kaggle CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cmd = ["kaggle", "kernels", "pull", kernel_ref, "-p", tmpdir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return ""

            for f in Path(tmpdir).iterdir():
                if f.suffix == '.ipynb':
                    with open(f, 'r', encoding='utf-8') as file:
                        notebook = json.load(file)
                    content = []
                    for cell in notebook.get('cells', []):
                        source = ''.join(cell.get('source', []))
                        content.append(source)
                    return '\n\n'.join(content)
                elif f.suffix == '.py':
                    return f.read_text(encoding='utf-8')
            return ""
        except Exception:
            return ""


def download_kernel_output(kernel_ref: str, output_dir: Path) -> Path | None:
    """Download kernel output (submission CSV) using kaggle CLI."""
    try:
        cmd = ["kaggle", "kernels", "output", kernel_ref, "-p", str(output_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        # Prefer submission.csv if it exists, otherwise first .csv
        csv_files = [f for f in output_dir.iterdir() if f.suffix == '.csv']
        for f in csv_files:
            if f.name == 'submission.csv':
                return f
        return csv_files[0] if csv_files else None
    except Exception:
        return None


def get_sample_submission(competition: str) -> pd.DataFrame | None:
    """Download sample_submission.csv from competition to get expected format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try common sample submission filenames
        for filename in ["sample_submission.csv", "sampleSubmission.csv"]:
            cmd = ["kaggle", "competitions", "download", "-c", competition,
                   "-f", filename, "-p", tmpdir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                fpath = Path(tmpdir) / filename
                if fpath.exists():
                    return pd.read_csv(fpath)
                # Might be zipped
                for f in Path(tmpdir).iterdir():
                    if f.suffix == '.csv':
                        return pd.read_csv(f)
                    elif f.suffix == '.zip':
                        import zipfile
                        with zipfile.ZipFile(f) as zf:
                            for name in zf.namelist():
                                if name.endswith('.csv'):
                                    with zf.open(name) as zfile:
                                        return pd.read_csv(zfile)
    return None


def extract_score_from_title(title: str) -> str:
    """Extract score from kernel title."""
    patterns = [
        r'([0-9]\.[0-9]{3,6})',
        r'([0-9])[_-]([0-9]{3,6})',
    ]
    for pattern in patterns:
        match = re.search(pattern, title.lower())
        if match:
            if len(match.groups()) == 2:
                return f"{match.group(1)}.{match.group(2)}"
            return match.group(1)
    return "-"

# %% cell 9

def test_llm_connection(api_key: str, model: str) -> tuple[bool, str]:
    """Test LLM connection with a simple request."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok'"}],
        "max_tokens": 10,
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=30
        )
        if response.status_code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif response.status_code == 402:
            return False, "Payment required - check OpenRouter credits (402)"
        elif response.status_code == 429:
            return False, "Rate limited (429)"
        elif response.status_code != 200:
            return False, f"HTTP {response.status_code}: {response.text[:100]}"
        response.raise_for_status()
        return True, ""
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"Unexpected error: {str(e)[:100]}"


def parse_llm_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if '```' in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    match = re.search(r'\{[^{}]*"is_blind_blend"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    is_blend = None
    if re.search(r'"is_blind_blend"\s*:\s*true', text, re.IGNORECASE):
        is_blend = True
    elif re.search(r'"is_blind_blend"\s*:\s*false', text, re.IGNORECASE):
        is_blend = False
    elif re.search(r'\bblind\s*blend\b', text, re.IGNORECASE) and re.search(r'\b(yes|true|is a)\b', text, re.IGNORECASE):
        is_blend = True
    elif re.search(r'\b(not|no|original)\b.*\bblind\s*blend\b', text, re.IGNORECASE):
        is_blend = False

    if is_blend is not None:
        conf_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', text, re.IGNORECASE)
        confidence = conf_match.group(1).lower() if conf_match else "medium"
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        reason = reason_match.group(1)[:100] if reason_match else "parsed from unstructured response"
        return {"is_blind_blend": is_blend, "confidence": confidence, "reason": reason}

    raise json.JSONDecodeError(f"Could not extract JSON from response: {text[:100]}...", text, 0)


def analyze_for_blind_blend(notebook_content: str, api_key: str, model: str, raise_on_error: bool = True) -> dict:
    """Use LLM to detect if notebook is a blind blend."""
    max_chars = 15000
    if len(notebook_content) > max_chars:
        notebook_content = notebook_content[:max_chars] + "\n...[truncated]..."

    prompt = f"""Analyze this Kaggle notebook and determine if it's a blind blend or ensemble of public submissions.

A BLIND BLEND notebook typically:
- Downloads or loads OTHER people's submission CSV files
- Averages, blends, or ensembles predictions from multiple public kernels
- Does NOT train its own model from scratch
- Uses terms like blend, ensemble submissions, and average predictions
- References other kernel names or URLs

An ORIGINAL notebook:
- Trains its own model(s) on the competition data
- May use an ensemble of ITS OWN models (that's fine)
- Does feature engineering and model training

example good orginal notebook: PlaygroundS6E3|Public|Baseline|V1

NOTEBOOK CONTENT:
{notebook_content}

Respond with ONLY a JSON object (no markdown):
{{"is_blind_blend": true/false, "confidence": "high/medium/low", "reason": "brief explanation"}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=60
        )
        response.raise_for_status()
        result_text = response.json()['choices'][0]['message']['content']
        return parse_llm_json_response(result_text)
    except requests.exceptions.Timeout:
        error_msg = "LLM request timeout"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return {"is_blind_blend": False, "confidence": "low", "reason": error_msg, "error": True}
    except requests.exceptions.HTTPError as e:
        error_msg = f"LLM API error: {e.response.status_code}"
        if raise_on_error:
            raise RuntimeError(f"{error_msg} - {e.response.text[:200]}")
        return {"is_blind_blend": False, "confidence": "low", "reason": error_msg, "error": True}
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response: {str(e)[:50]}"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return {"is_blind_blend": False, "confidence": "low", "reason": error_msg, "error": True}
    except Exception as e:
        error_msg = f"LLM error: {str(e)[:80]}"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return {"is_blind_blend": False, "confidence": "low", "reason": error_msg, "error": True}

# %% cell 10

def download_and_filter_submissions(
    competition: str,
    top_n: int,
    best: str,
    output_dir: Path,
    llm_filter: bool = False,
    llm_model: str = "google/gemini-2.5-flash"
) -> tuple[list[pd.DataFrame], list[Path]]:
    """Download top kernel submissions and optionally filter blind blends."""
    api_key = os.getenv("OPENROUTER_API_KEY") if llm_filter else None
    if llm_filter and not api_key:
        print("Warning: LLM filter requested but OPENROUTER_API_KEY not set. Skipping LLM filter.")
        llm_filter = False

    if llm_filter:
        print(f"\nLLM Filter enabled:")
        print(f"  Model: {llm_model}")
        print(f"  Testing connection...", end=" ", flush=True)
        success, error = test_llm_connection(api_key, llm_model)
        if not success:
            print(f"FAILED: {error}")
            print("  Continuing without LLM filter")
            llm_filter = False
        else:
            print("OK")

    print(f"\nFetching kernels for {competition} (best={best})...")
    kernels = get_kernels_list(competition, page_size=top_n * 3, best=best)
    print(f"Found {len(kernels)} kernels")

    if not kernels:
        print("No kernels found. Check competition slug.")
        return [], []

    output_dir.mkdir(parents=True, exist_ok=True)

    submissions = []
    files = []
    reference_cols = None
    reference_rows = None
    notebook_results = []

    # Download sample submission to establish expected format
    print("Downloading sample submission for reference format...")
    sample_sub = get_sample_submission(competition)
    if sample_sub is not None and len(sample_sub.columns) == 2:
        reference_cols = list(sample_sub.columns)
        reference_rows = len(sample_sub)
        print(f"  Reference: {reference_rows} rows, columns: {reference_cols}")
    else:
        print("  Could not get sample submission, will use first download as reference")

    print(f"\n{'='*80}")
    print(f"{'#':<3} {'Notebook':<40} {'Score':<10} {'Status'}")
    print("-" * 80)

    downloaded = 0
    for kernel in kernels:
        if downloaded >= top_n:
            break

        kernel_ref = kernel.get('ref', '')
        kernel_title = kernel.get('title', kernel_ref)
        kernel_name = kernel_ref.split('/')[-1] if '/' in kernel_ref else kernel_ref
        score = extract_score_from_title(kernel_title)

        result = {"name": kernel_name, "score": score, "status": ""}

        # LLM filter check
        if llm_filter:
            source = get_notebook_source(kernel_ref)
            if source:
                try:
                    analysis = analyze_for_blind_blend(source, api_key, llm_model, raise_on_error=True)
                    if analysis.get("is_blind_blend", False):
                        confidence = analysis.get("confidence", "?")
                        reason = analysis.get("reason", "")[:30]
                        result["status"] = f"BLIND BLEND ({confidence})"
                        notebook_results.append(result)
                        print(f"   {kernel_name[:40]:<40} {score:<10} BLIND BLEND ({confidence}: {reason})")
                        continue
                except RuntimeError as e:
                    print(f"\n   ERROR analyzing {kernel_name}: {e}")
                    raise

        # Download the submission output
        kernel_output_dir = output_dir / f"sub_{downloaded+1:02d}_{kernel_name[:30]}"
        kernel_output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = download_kernel_output(kernel_ref, kernel_output_dir)

        if csv_path:
            try:
                df = pd.read_csv(csv_path)

                if len(df.columns) != 2:
                    result["status"] = "SKIP (columns)"
                    notebook_results.append(result)
                    print(f"   {kernel_name[:40]:<40} {score:<10} SKIP (columns)")
                    shutil.rmtree(kernel_output_dir, ignore_errors=True)
                    continue

                if reference_cols is None:
                    reference_cols = list(df.columns)
                    reference_rows = len(df)

                if len(df) != reference_rows:
                    result["status"] = "SKIP (rows)"
                    notebook_results.append(result)
                    print(f"   {kernel_name[:40]:<40} {score:<10} SKIP (rows)")
                    shutil.rmtree(kernel_output_dir, ignore_errors=True)
                    continue

                df.columns = reference_cols

                new_path = output_dir / f"sub_{downloaded+1:02d}_{kernel_name[:30]}.csv"
                df.to_csv(new_path, index=False)
                submissions.append(df)
                files.append(new_path)
                downloaded += 1
                result["status"] = "OK"
                notebook_results.append(result)
                print(f"{downloaded:2d}. {kernel_name[:40]:<40} {score:<10} OK")

                shutil.rmtree(kernel_output_dir, ignore_errors=True)
            except Exception:
                result["status"] = "FAILED"
                notebook_results.append(result)
                print(f"   {kernel_name[:40]:<40} {score:<10} FAILED")
                shutil.rmtree(kernel_output_dir, ignore_errors=True)
        else:
            result["status"] = "NO OUTPUT"
            notebook_results.append(result)
            print(f"   {kernel_name[:40]:<40} {score:<10} NO OUTPUT")
            shutil.rmtree(kernel_output_dir, ignore_errors=True)

    blind_count = sum(1 for r in notebook_results if "BLIND BLEND" in r["status"])
    print("-" * 80)
    print(f"Downloaded: {len(submissions)} | Blind blends filtered: {blind_count} | Total checked: {len(notebook_results)}")
    print(f"{'='*80}\n")

    return submissions, files

# %% cell 11

def get_target_matrix(submissions: list[pd.DataFrame], target_col: str) -> np.ndarray:
    return np.column_stack([df[target_col].values for df in submissions])


def compute_correlation_matrix(targets: np.ndarray) -> np.ndarray:
    return np.corrcoef(targets.T)


def is_blend_submission(filename: str) -> bool:
    blend_keywords = [
        'blend', 'ensemble', 'stack', 'mix', 'combine', 'merge', 'avg',
        'mean', 'median', 'weighted', 'meta', 'final'
    ]
    name_lower = filename.lower()
    return any(kw in name_lower for kw in blend_keywords)


def filter_base_models(submissions, files, exclude_blends=True):
    if not exclude_blends:
        return submissions, files, []
    filtered_subs, filtered_files, excluded = [], [], []
    for sub, f in zip(submissions, files):
        if is_blend_submission(f.name):
            excluded.append(f.name)
        else:
            filtered_subs.append(sub)
            filtered_files.append(f)
    return filtered_subs, filtered_files, excluded


def filter_incompatible_scales(submissions, files, target_col, threshold=0.5):
    means = np.array([df[target_col].mean() for df in submissions])
    stds = np.array([df[target_col].std() for df in submissions])
    median_mean, median_std = np.median(means), np.median(stds)
    filtered_subs, filtered_files, excluded = [], [], []
    for sub, f, m, s in zip(submissions, files, means, stds):
        mean_diff = abs(m - median_mean) / (abs(median_mean) + 1e-10)
        std_diff = abs(s - median_std) / (abs(median_std) + 1e-10)
        if mean_diff <= threshold and std_diff <= threshold:
            filtered_subs.append(sub)
            filtered_files.append(f)
        else:
            excluded.append(f.name)
    return filtered_subs, filtered_files, excluded

# %% cell 12

def greedy_diverse_selection(targets, corr_matrix, files, max_subs=10, min_corr_threshold=0.995):
    n_subs = targets.shape[1]
    if n_subs <= max_subs:
        return list(range(n_subs)), []

    selected = [0]
    remaining = set(range(1, n_subs))
    selection_log = [{"idx": 0, "file": files[0].name, "max_corr": None, "reason": "seed (best LB)"}]

    while len(selected) < max_subs and remaining:
        best_idx = None
        best_max_corr = 1.0
        for idx in remaining:
            max_corr = max(corr_matrix[idx, sel] for sel in selected)
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_idx = idx
        if best_idx is None or best_max_corr > min_corr_threshold:
            selection_log.append({
                "idx": None, "file": None, "max_corr": best_max_corr,
                "reason": f"stopped: all remaining corr > {min_corr_threshold}"
            })
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        selection_log.append({
            "idx": best_idx, "file": files[best_idx].name,
            "max_corr": best_max_corr, "reason": "lowest max-correlation"
        })
    return selected, selection_log


def cluster_based_selection(targets, corr_matrix, files, n_clusters=5):
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip((dist_matrix + dist_matrix.T) / 2, 0, None)

    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    selected = []
    cluster_info = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            best_in_cluster = cluster_indices[0]
            selected.append(best_in_cluster)
            cluster_info.append({
                "cluster_id": cluster_id,
                "size": len(cluster_indices),
                "selected_idx": best_in_cluster,
                "selected_file": files[best_in_cluster].name,
                "members": [files[i].name for i in cluster_indices]
            })
    return selected, cluster_info, cluster_labels


def top_n_selection(targets, files, n=10):
    return list(range(min(n, targets.shape[1])))

# %% cell 13

def ensemble_mean(targets, indices):
    return targets[:, indices].mean(axis=1)


def ensemble_median(targets, indices):
    return np.median(targets[:, indices], axis=1)


def ensemble_trimmed_mean(targets, indices, trim=0.1):
    from scipy import stats
    return stats.trim_mean(targets[:, indices], trim, axis=1)


def ensemble_geometric_mean(targets, indices):
    selected = np.clip(targets[:, indices], 1e-10, 1 - 1e-10)
    return np.exp(np.mean(np.log(selected), axis=1))


def ensemble_log_odds(targets, indices):
    selected = np.clip(targets[:, indices], 1e-10, 1 - 1e-10)
    log_odds = np.log(selected / (1 - selected))
    avg_log_odds = np.mean(log_odds, axis=1)
    return 1 / (1 + np.exp(-avg_log_odds))


def ensemble_rank_average(targets, indices):
    from scipy import stats
    selected = targets[:, indices]
    n_samples = selected.shape[0]
    ranks = np.zeros_like(selected)
    for i in range(selected.shape[1]):
        ranks[:, i] = stats.rankdata(selected[:, i]) / n_samples
    return np.mean(ranks, axis=1)


def ensemble_power_mean(targets, indices, p=2):
    selected = np.clip(targets[:, indices], 1e-10, 1)
    return np.power(np.mean(np.power(selected, p), axis=1), 1/p)


def ensemble_rank_weighted(targets, indices, main_weights=None, position_weights=None, asc_ratio=0.3):
    selected = targets[:, indices]
    n_samples, n_subs = selected.shape

    if main_weights is None:
        main_weights = np.array([0.5] + [0.5 / (n_subs - 1)] * (n_subs - 1))
        main_weights = main_weights[:n_subs]
        main_weights /= main_weights.sum()

    if position_weights is None:
        position_weights = np.zeros(n_subs)
        if n_subs >= 2:
            position_weights[0] = 0.07
            position_weights[-1] = -0.07
            for i in range(1, n_subs - 1):
                position_weights[i] = 0.07 - (0.14 * i / (n_subs - 1))

    def blend_with_sort(ascending):
        result = np.zeros(n_samples)
        for row_idx in range(n_samples):
            row_values = selected[row_idx, :]
            sorted_indices = np.argsort(row_values)
            if not ascending:
                sorted_indices = sorted_indices[::-1]
            weighted_sum = 0.0
            total_weight = 0.0
            for position, sub_idx in enumerate(sorted_indices):
                effective_weight = main_weights[sub_idx] + position_weights[position]
                effective_weight = max(0.001, effective_weight)
                weighted_sum += row_values[sub_idx] * effective_weight
                total_weight += effective_weight
            result[row_idx] = weighted_sum / total_weight
        return result

    asc_result = blend_with_sort(ascending=True)
    desc_result = blend_with_sort(ascending=False)
    return asc_ratio * asc_result + (1 - asc_ratio) * desc_result


def hill_climbing_weights(targets, indices, n_iterations=100, step_size=0.05):
    selected_targets = targets[:, indices]
    n_selected = len(indices)
    weights = np.ones(n_selected) / n_selected
    reference = targets.mean(axis=1)

    def score(w):
        pred = (selected_targets * w).sum(axis=1)
        return np.corrcoef(pred, reference)[0, 1]

    best_score = score(weights)
    for _ in range(n_iterations):
        for i in range(n_selected):
            for delta in [step_size, -step_size]:
                new_weights = weights.copy()
                new_weights[i] += delta
                new_weights = np.clip(new_weights, 0.01, 1.0)
                new_weights /= new_weights.sum()
                new_score = score(new_weights)
                if new_score > best_score:
                    weights = new_weights
                    best_score = new_score

    result = (selected_targets * weights).sum(axis=1)
    return result, weights.tolist()


def ensemble_rank_weighted_optuna(targets, indices, n_trials=100, n_folds=5, seed=42):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    selected = targets[:, indices]
    n_samples, n_subs = selected.shape
    reference = targets.mean(axis=1)

    np.random.seed(seed)
    fold_indices = np.random.randint(0, n_folds, size=n_samples)

    def objective(trial):
        raw_weights = [trial.suggest_float(f"w_{i}", 0.01, 1.0) for i in range(n_subs)]
        main_weights = np.array(raw_weights)
        main_weights /= main_weights.sum()

        position_weights = np.array([
            trial.suggest_float(f"pos_{i}", -0.15, 0.15) for i in range(n_subs)
        ])
        asc_ratio = trial.suggest_float("asc_ratio", 0.0, 1.0)

        cv_scores = []
        for fold in range(n_folds):
            val_mask = fold_indices == fold
            val_pred = ensemble_rank_weighted(
                selected[val_mask, :], list(range(n_subs)),
                main_weights, position_weights, asc_ratio
            )
            val_ref = reference[val_mask]
            corr = np.corrcoef(val_pred, val_ref)[0, 1]
            cv_scores.append(corr)
        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    main_weights = np.array([best_params[f"w_{i}"] for i in range(n_subs)])
    main_weights /= main_weights.sum()
    position_weights = np.array([best_params[f"pos_{i}"] for i in range(n_subs)])
    asc_ratio = best_params["asc_ratio"]

    result = ensemble_rank_weighted(selected, list(range(n_subs)), main_weights, position_weights, asc_ratio)
    return result, {
        "main_weights": main_weights.tolist(),
        "position_weights": position_weights.tolist(),
        "asc_ratio": asc_ratio,
        "cv_score": study.best_value,
    }

# %% cell 14

# Step 1: Download submissions
print(f"Kaggle mode: downloading from {COMPETITION}")
submissions, files = download_and_filter_submissions(
    competition=COMPETITION,
    top_n=TOP_N,
    best=BEST,
    output_dir=SUBS_DIR,
    llm_filter=LLM_FILTER,
    llm_model=LLM_MODEL,
)

assert submissions, "No submissions found!"

# Step 2: Detect columns
id_col, target_col = submissions[0].columns[0], submissions[0].columns[1]

# Step 3: Filter incompatible scales
submissions, files, excluded_scale = filter_incompatible_scales(submissions, files, target_col)

# Step 4: Optionally filter blends by name
excluded_blends = []
if EXCLUDE_BLENDS:
    submissions, files, excluded_blends = filter_base_models(submissions, files, exclude_blends=True)
    if excluded_blends:
        print(f"\nExcluded {len(excluded_blends)} blend submissions")

assert len(submissions) >= 2, "Not enough submissions after filtering!"
print(f"\nUsing {len(submissions)} submissions after filtering")

# Step 5: Correlation analysis
targets = get_target_matrix(submissions, target_col)
corr_matrix = compute_correlation_matrix(targets)
upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

print(f"\nCorrelation summary:")
print(f"  Min: {upper_tri.min():.4f}")
print(f"  Max: {upper_tri.max():.4f}")
print(f"  Mean: {upper_tri.mean():.4f}")

# Local scores
mean_predictions = targets.mean(axis=1)
local_scores = [np.corrcoef(targets[:, i], mean_predictions)[0, 1] for i in range(targets.shape[1])]

print(f"\n{'='*80}")
print(f"{'#':<3} {'Submission':<45} {'Local Score':<12} {'Diversity'}")
print("-" * 80)

sorted_indices = np.argsort(local_scores)[::-1]
for rank, i in enumerate(sorted_indices):
    mean_corr_others = (corr_matrix[i, :].sum() - 1) / (len(files) - 1)
    diversity = 1 - mean_corr_others
    print(f"{rank+1:2d}. {files[i].name[:45]:<45} {local_scores[i]:.6f}    {diversity:.4f}")

print("-" * 80)
print(f"Local Score = correlation with ensemble mean (higher = more consensus)")
print(f"Diversity = 1 - mean correlation with others (higher = more unique)")
print(f"{'='*80}")

if SHOW_CORR:
    print("\nCorrelation matrix (first 10):")
    n_show = min(10, len(files))
    for i in range(n_show):
        row = " ".join(f"{corr_matrix[i,j]:.3f}" for j in range(n_show))
        print(f"  {files[i].name[:20]:20s} | {row}")

# %% cell 15

cluster_info = None
selection_log = None

if STRATEGY == "greedy":
    selected_indices, selection_log = greedy_diverse_selection(
        targets, corr_matrix, files, MAX_SUBS, MIN_CORR
    )
    print(f"\nGreedy diverse selection (target: {MAX_SUBS}):")
    for entry in selection_log:
        if entry['idx'] is not None:
            corr_str = f"(max_corr={entry['max_corr']:.4f})" if entry['max_corr'] else "(seed)"
            print(f"  [{selection_log.index(entry)+1}] {entry['file']} {corr_str}")
        else:
            print(f"  {entry['reason']}")

elif STRATEGY == "cluster":
    selected_indices, cluster_info, _ = cluster_based_selection(
        targets, corr_matrix, files, MAX_SUBS
    )
    print(f"\nCluster-based selection ({MAX_SUBS} clusters):")
    for c in cluster_info:
        print(f"  Cluster {c['cluster_id']}: {c['selected_file']} (from {c['size']} submissions)")

else:  # top
    selected_indices = top_n_selection(targets, files, MAX_SUBS)
    print(f"\nTop-{MAX_SUBS} selection:")
    for i, idx in enumerate(selected_indices):
        print(f"  [{i+1}] {files[idx].name}")

print(f"\nSelected {len(selected_indices)} submissions")

# %% cell 16

print(f"\nEnsemble method: {ENSEMBLE_METHOD}")
hill_weights = None

if ENSEMBLE_METHOD == "mean":
    result = ensemble_mean(targets, selected_indices)
elif ENSEMBLE_METHOD == "median":
    result = ensemble_median(targets, selected_indices)
elif ENSEMBLE_METHOD == "trimmed":
    result = ensemble_trimmed_mean(targets, selected_indices)
elif ENSEMBLE_METHOD == "geomean":
    result = ensemble_geometric_mean(targets, selected_indices)
elif ENSEMBLE_METHOD == "logodds":
    result = ensemble_log_odds(targets, selected_indices)
elif ENSEMBLE_METHOD == "rank":
    result = ensemble_rank_average(targets, selected_indices)
elif ENSEMBLE_METHOD == "power":
    result = ensemble_power_mean(targets, selected_indices, p=2)
elif ENSEMBLE_METHOD == "hill":
    result, hill_weights = hill_climbing_weights(targets, selected_indices)
    print("  Optimized weights:")
    for idx, w in zip(selected_indices, hill_weights):
        print(f"    {files[idx].name}: {w:.4f}")
elif ENSEMBLE_METHOD == "rankweight":
    result = ensemble_rank_weighted(targets, selected_indices)
    print("  Using default rank-weighted params")
elif ENSEMBLE_METHOD == "rankweight_optuna":
    print(f"  Optimizing with {OPTUNA_TRIALS} Optuna trials...")
    result, optuna_params = ensemble_rank_weighted_optuna(
        targets, selected_indices, n_trials=OPTUNA_TRIALS
    )
    if optuna_params:
        print(f"  Best CV score: {optuna_params.get('cv_score', 'N/A'):.6f}")
        print(f"  Asc/Desc ratio: {optuna_params.get('asc_ratio', 0.5):.3f}")
        print("  Optimized main weights:")
        for idx, w in zip(selected_indices, optuna_params.get('main_weights', [])):
            print(f"    {files[idx].name}: {w:.4f} ({w*100:.1f}%)")
        hill_weights = optuna_params.get('main_weights', [])

# %% cell 17

# Create submission CSV
output_df = submissions[0][[id_col]].copy()
output_df[target_col] = result
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved: {OUTPUT_PATH}")
print(f"Stats: min={result.min():.4f}, max={result.max():.4f}, "
      f"mean={result.mean():.4f}, std={result.std():.4f}")

# Clean up intermediate files
if SUBS_DIR.exists():
    shutil.rmtree(SUBS_DIR)
    print(f"Cleaned up temp directory: {SUBS_DIR}")

# Verify only submission.csv remains
remaining_files = list(WORK_DIR.glob("*"))
print(f"\nFiles in {WORK_DIR}:")
for f in remaining_files:
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Submit to Kaggle
import time

message = f"Smart ensemble: {STRATEGY} + {ENSEMBLE_METHOD} (top {TOP_N}, max {MAX_SUBS})"
cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(OUTPUT_PATH), "-m", message]
print(f"\nSubmitting to {COMPETITION}...")
sub_result = subprocess.run(cmd, capture_output=True, text=True)
if sub_result.returncode != 0:
    print(f"Submit failed: {sub_result.stderr}")
else:
    print(f"Submitted! Message: {message}")
    print("Waiting for score...", flush=True)
    for attempt in range(12):
        time.sleep(5)
        cmd = ["kaggle", "competitions", "submissions", "-c", COMPETITION, "-v"]
        check = subprocess.run(cmd, capture_output=True, text=True)
        if check.returncode == 0:
            lines = check.stdout.strip().split('\n')
            if len(lines) >= 2:
                headers = lines[0].split(',')
                values = lines[1].split(',')
                if len(values) >= len(headers):
                    sub = dict(zip(headers, values))
                    score = sub.get('publicScore', '')
                    if score and score != 'None':
                        print(f"\nPUBLIC LB SCORE: {score}")
                        break
                    elif sub.get('status', '') == 'error':
                        print(f"\nSubmission error: {sub.get('errorDescription', 'Unknown')}")
                        break
        print(f"  Waiting... ({(attempt+1)*5}s)")
    else:
        print("\nTimeout waiting for score. Check Kaggle submissions page.")

