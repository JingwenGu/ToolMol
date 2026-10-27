from __future__ import print_function
import os

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='molleo_multi')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mol_lm', type=str, default=None, choices=[None, "BioT5", "MoleculeSTM", "GPT-4", "ToolMol"])
    parser.add_argument('--llm_model', type=str, default='gpt-4o', help='Model name/id passed to ToolMol\'s agent - an OpenAI-compatible model name for --llm_backend api, or a local/HF model id (e.g. Qwen/Qwen3-4B-Instruct-2507) for --llm_backend transformers. Needs a large context window since get_ligand_structure dumps can be verbose - gpt-4\'s 8192-token limit is too small.')
    parser.add_argument('--llm_backend', type=str, default='api', choices=['api', 'responses', 'claude', 'transformers', 'stub'], help='"api" (default) talks to any OpenAI-compatible chat-completions endpoint via --llm_base_url - the OpenAI API itself, or a locally-hosted server such as vLLM (see multi_objective/main/toolmol/serve_local_vllm.sh) or Ollama. "responses" talks to the same kind of server\'s /v1/responses endpoint instead - required for gpt-oss models: confirmed against vLLM 0.28.0 that gpt-oss\'s tool_calls never come back populated over chat completions (the intent to call a tool only shows up in an unstructured reasoning string), while /v1/responses (OpenAI\'s native Harmony-format endpoint) surfaces them correctly. "claude" runs no model at all: each agent turn is spooled as JSON to $TOOLMOL_CLAUDE_SPOOL and the operator answers it by hand, acting as the agent directly (see agent.py\'s _create_claude). "transformers" loads --llm_model in-process with Hugging Face transformers instead, for local inference with no server to run (needs a GPU with enough VRAM for the model, and the transformers/torch/accelerate packages)')
    parser.add_argument('--llm_base_url', type=str, default=None, help='Override API base URL for --llm_backend api, e.g. to point at a GPT-OSS-120B-compatible endpoint or a local vLLM/Ollama server (http://localhost:8000/v1)')
    parser.add_argument('--llm_api_key_env', type=str, default='OPENAI_API_KEY', help='Env var name holding the API key for --llm_base_url, e.g. DEEPINFRA_API_KEY when pointed at a non-OpenAI provider. Unused for --llm_backend transformers. Local servers such as vLLM ignore the key value by default, but the env var still needs to be set to *something* (e.g. export OPENAI_API_KEY=not-needed) since the client requires one')
    parser.add_argument('--llm_extra_body', type=str, default=None, help='JSON string passed as extra_body to the chat completions call, e.g. \'{"provider": {"only": ["groq"]}}\' to pin OpenRouter to a specific underlying provider. Unused for --llm_backend transformers')
    parser.add_argument('--llm_device_map', type=str, default='auto', help='device_map passed to transformers.AutoModelForCausalLM.from_pretrained for --llm_backend transformers')
    parser.add_argument('--llm_max_new_tokens', type=int, default=1024, help='max_new_tokens per generation step for --llm_backend transformers')
    parser.add_argument('--llm_system_prompt_suffix', type=str, default=None, help='Text appended to SYSTEM_PROMPT verbatim. Needed for models/providers whose Chat Completions response has no reasoning_content/reasoning field (e.g. OpenAI gpt-5-family) - without an explicit instruction, those models return content=null alongside a tool call with no visible rationale at all, making failure-mode analysis of the run impossible.')
    parser.add_argument('--few_shot_file', type=str, default=None, help='Path to a JSON message-list file (see agent.py\'s load_few_shot_messages) prepended verbatim before the live query on every agent call. Off by default (None) - every existing invocation is unaffected. Toggle on for a specific experiment by pointing this at a generated few-shot library.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a previously saved results_*.yaml (from ToolMol\'s save_result) to resume from - reconstructs the starting population by re-running select_pareto_front() over the full saved set (not a top-N-by-score truncation) and carries its oracle-call history forward, so max_oracle_calls counts from where the earlier run left off instead of from 0. Also loads the sibling pareto_snapshots_*.yaml in the same directory, if present, so the every-10-generation snapshot history accumulates continuously across chained runs instead of each one starting empty and overwriting the last. Only supported by --mol_lm ToolMol. Useful for chaining several short (fast-to-schedule) jobs into one long search.')
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--max_obj', nargs="+", default=["jnk3"]) ### 
    parser.add_argument('--min_obj', nargs="+", default=["sa"]) ### 
    parser.add_argument('--task_mode', type=str, default='1')
    # --- C-MuMO arm ---
    parser.add_argument('--improve', nargs='+', default=None,
                        help='properties to improve (Pareto objectives). Direction comes '
                             'from the property table, not from this flag.')
    parser.add_argument('--maintain', nargs='+', default=[],
                        help='properties to hold inside their C-MuMO threshold (hinge '
                             'objectives, flat on the plateau).')
    parser.add_argument('--combination', type=str, default=None,
                        help="a C-MuMO task code (BPQ, ELQ, ACEP, BDPQ, BCMQ, HLMPQ). The "
                             "two lowest-occupancy properties become the improve axes, the "
                             "rest maintain, per the calibration. Overrides "
                             "--improve/--maintain.")
    parser.add_argument('--reject_at', type=float, default=5.0,
                        help='drop offspring violating any maintain axis by more than this '
                             'many multiples of M, before the Pareto front (design 3.2). '
                             'Defaults to the clamp C, where the hinge already scores 0.')
    parser.add_argument('--population_floor', type=int, default=0,
                        help='if the Pareto front collapses below 2, top it back up to this '
                             'many molecules with the next-best by Phi. Off by default, '
                             'since it changes the inherited survival semantics; 4 is a '
                             'sensible value whenever one molecule can dominate the whole '
                             'population and end the search.')
    parser.add_argument('--front_cap', type=int, default=40,
                        help='cap candidates entering the front, highest Phi first (design '
                             '3.3). A safety valve: calibration measured fronts at 6-10 of '
                             '95, so this should never bind in a healthy run.')
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_dir', default="./results")
    args = parser.parse_args()


    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones

    if args.method == 'molleo_multi':
        from main.molleo_multi.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'molleo_multi_pareto':
        from main.molleo_multi_pareto.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'toolmol':
        from main.toolmol.run import GB_GA_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")


    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main


    # The C-MuMO objective set. Every property goes in max_obj because rescaled() already
    # returns 1 = best for lower-is-better properties too; putting one in min_obj would
    # invert it twice. The improve/maintain split is the real configuration and reaches the
    # objectives module (and any child process) through configure().
    from main.toolmol import objectives as cmumo_objectives
    from main.toolmol.combinations import resolve_combination

    if args.combination:
        improve, maintain = resolve_combination(args.combination)
        print(f"combination {args.combination}: improve={improve} maintain={maintain}")
    else:
        improve = args.improve if args.improve is not None else ['QED']
        maintain = list(args.maintain)
    overlap = sorted(set(improve) & set(maintain))
    if overlap:
        raise SystemExit(f"properties in both --improve and --maintain: {overlap}")
    args.improve, args.maintain = list(improve), list(maintain)
    cmumo_objectives.configure(args.improve, args.maintain)
    args.max_obj = args.improve + args.maintain
    args.min_obj = []
    print(f'improve (Pareto axes): {args.improve}')
    print(f'maintain (hinge axes): {args.maintain}')

    if args.patience < args.max_oracle_calls:
        # Default is 5, and the stop check fires when the mean buffer score improves by
        # less than 1e-3 for that many consecutive generations - which on a Pareto front
        # flattens long before the budget runs out. This truncated run 1 of both surrogate
        # arms, so it is an error here rather than a footnote.
        raise SystemExit(
            f"--patience {args.patience} is below --max_oracle_calls "
            f"{args.max_oracle_calls}; set them equal so the budget is the only stop "
            f"(CMUMO_PORT_DESIGN.md section 7).")

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

    optimizer = Optimizer(args=args)
    print(config_default)
                
    for seed in args.seed:
        print('seed', seed)
        optimizer.optimize(config=config_default, seed=seed)


    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

