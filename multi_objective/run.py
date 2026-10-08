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
    parser.add_argument('--llm_backend', type=str, default='api', choices=['api', 'responses', 'transformers'], help='"api" (default) talks to any OpenAI-compatible chat-completions endpoint via --llm_base_url - the OpenAI API itself, or a locally-hosted server such as vLLM (see multi_objective/main/toolmol/serve_local_vllm.sh) or Ollama. "responses" talks to the same kind of server\'s /v1/responses endpoint instead - required for gpt-oss models: confirmed against vLLM 0.28.0 that gpt-oss\'s tool_calls never come back populated over chat completions (the intent to call a tool only shows up in an unstructured reasoning string), while /v1/responses (OpenAI\'s native Harmony-format endpoint) surfaces them correctly. "transformers" loads --llm_model in-process with Hugging Face transformers instead, for local inference with no server to run (needs a GPU with enough VRAM for the model, and the transformers/torch/accelerate packages)')
    parser.add_argument('--llm_base_url', type=str, default=None, help='Override API base URL for --llm_backend api, e.g. to point at a GPT-OSS-120B-compatible endpoint or a local vLLM/Ollama server (http://localhost:8000/v1)')
    parser.add_argument('--llm_api_key_env', type=str, default='OPENAI_API_KEY', help='Env var name holding the API key for --llm_base_url, e.g. DEEPINFRA_API_KEY when pointed at a non-OpenAI provider. Unused for --llm_backend transformers. Local servers such as vLLM ignore the key value by default, but the env var still needs to be set to *something* (e.g. export OPENAI_API_KEY=not-needed) since the client requires one')
    parser.add_argument('--llm_extra_body', type=str, default=None, help='JSON string passed as extra_body to the chat completions call, e.g. \'{"provider": {"only": ["groq"]}}\' to pin OpenRouter to a specific underlying provider. Unused for --llm_backend transformers')
    parser.add_argument('--llm_device_map', type=str, default='auto', help='device_map passed to transformers.AutoModelForCausalLM.from_pretrained for --llm_backend transformers')
    parser.add_argument('--llm_max_new_tokens', type=int, default=1024, help='max_new_tokens per generation step for --llm_backend transformers')
    parser.add_argument('--llm_system_prompt_suffix', type=str, default=None, help='Text appended to SYSTEM_PROMPT verbatim. Needed for models/providers whose Chat Completions response has no reasoning_content/reasoning field (e.g. OpenAI gpt-5-family) - without an explicit instruction, those models return content=null alongside a tool call with no visible rationale at all, making failure-mode analysis of the run impossible.')
    parser.add_argument('--few_shot_file', type=str, default=None, help='Path to a JSON message-list file (see agent.py\'s load_few_shot_messages) prepended verbatim before the live query on every agent call. Off by default (None) - every existing invocation is unaffected. Toggle on for a specific experiment by pointing this at a generated few-shot library. Mutually exclusive with --online_fewshot.')
    parser.add_argument('--online_fewshot', action='store_true', help='Maintain the few-shot buffer live during the run instead of loading a fixed file (see main/toolmol/online_fewshot.py): every --online_fewshot_update_every episodes, sample --online_fewshot_sample_size recently-encountered starting states in a background thread, run a beam=--online_fewshot_beam non-myopic 3-step search on each, post-rationalize the winners, and splice them into a fixed --online_fewshot_buffer_size FIFO buffer (oldest evicted first). Mutually exclusive with --few_shot_file. Uses --llm_model/--llm_api_key_env for the post-rationalization calls.')
    parser.add_argument('--online_fewshot_update_every', type=int, default=100, help='Episodes per window; one beam-search update cycle is launched per window, none skipped. The episode loop runs alongside a cycle for one window, but if that cycle is still running when the next window closes, the loop pauses until it finishes (backpressure), so a cycle slower than a window costs wall-clock time rather than dropped updates.')
    parser.add_argument('--online_fewshot_sample_size', type=int, default=2, help='New examples the online buffer tries to add per update cycle. Each cycle makes at most max(3x this, 6) beam-search attempts, so cycle time scales roughly linearly with it. Values near --online_fewshot_buffer_size replace most or all of the buffer in a single cycle.')
    parser.add_argument('--online_fewshot_beam', type=int, default=5, help='Beam width for the online buffer\'s 3-step search - deliberately much smaller than the offline fewshot_gen pipeline\'s default of 30, since this runs in the background during a live run and has to stay well under --online_fewshot_update_every episodes\' worth of wall time.')
    parser.add_argument('--online_fewshot_buffer_size', type=int, default=10, help='Max examples kept in the online few-shot buffer (FIFO eviction).')
    parser.add_argument('--online_fewshot_workers', type=int, default=8, help='Worker processes in the online buffer\'s persistent search pool.')
    parser.add_argument('--online_fewshot_diversity_floor', type=float, default=0.6, help='A candidate is rejected if adding it would drop the buffer\'s own diversity (tdc.Evaluator(\'Diversity\')) below this fraction of the live population\'s current diversity.')
    parser.add_argument('--online_fewshot_max_repeats', type=int, default=2, help='A candidate is rejected if any step in its sequence (by (tool, content) identity - e.g. add_atom(F) vs add_atom(C) count as different, but atom indices are ignored) would be the same move already used by --online_fewshot_max_repeats or more existing buffer examples at that same step position. Must be >= 2 per the finding that the small, finite action vocabulary makes single-example overlap common and expected, not evidence of redundancy.')
    parser.add_argument('--online_fewshot_seed_file', type=str, default=None, help='Path to a fewshot_gen-shaped JSON file (e.g. main/toolmol/fewshot_gen/fewshot_examples_3step.json) whose path_steps/per_step_reasoning entries warm-start the online buffer instead of starting empty. Avoids the cold-start ramp where the buffer holds fewer than its full complement of examples (or none at all) for a large fraction of the run.')
    parser.add_argument('--online_fewshot_load_buffer', type=str, default=None, help='Path to an online_fewshot_buffer.json written by an earlier run (every --online_fewshot run saves one to its --output_dir after each cycle) to start the buffer from, in place of --online_fewshot_seed_file. Restores the exact few-shot messages; the molecule population itself still comes from --resume_from. Mutually exclusive with --online_fewshot_seed_file.')
    parser.add_argument('--online_fewshot_inject', action='store_true', help='Also fold each newly-accepted buffer entry\'s final (near-optimal) molecule directly into the live GA population, competing for a Pareto-front slot alongside the LLM\'s own offspring - not just serving as future few-shot text. Off by default: every existing run is unaffected unless this is passed. Changes what the run measures (population quality is no longer purely a function of the LLM\'s own edits), so avg_top1/top10/top100 from a run with this enabled aren\'t directly comparable to a run without it.')
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


    print(f'Optimizing oracle function: {args.max_obj}')
    print(f'Optimizing oracle function: {args.min_obj}')

    if args.max_obj == ['jnk3', 'qed'] and args.min_obj == ['sa']:
        args.task_mode = '1'
    elif args.max_obj == ['gsk3', 'qed'] and args.min_obj == ['sa']:
        args.task_mode = '2'
    elif args.max_obj == ['jnk3', 'qed'] and args.min_obj == ['sa', 'gsk3b', 'drd2']:
        args.task_mode = '3'
    else:
        print('Undefined Task!')

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

    optimizer = Optimizer(args=args)
    print(config_default)
                
    for seed in args.seed:
        print('seed', seed)
        optimizer.optimize(config=config_default, seed=seed)

    if getattr(optimizer, "online_buffer", None) is not None:
        optimizer.online_buffer.shutdown()

    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

