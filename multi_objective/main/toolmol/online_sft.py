"""Online SFT: the fine-tuning analogue of online few-shot (online_fewshot.py).

Same buffer, same beam-search-plus-post-rationalization example builder, same acceptance
gates - OnlineSFTManager wraps an existing OnlineFewShotBuffer and reuses it unchanged for
all of that. The only thing it replaces is how the buffer's contents reach the model: instead
of exposing entries as few-shot conversation history prepended to every prompt
(buffer.few_shot_messages), it periodically fine-tunes the base model on the buffer's current
entries and swaps the serving endpoint over to the result - see main/toolmol/run.py's
_optimize(), which calls maybe_finetune() instead of setting mol_lm.few_shot_messages when
--online_fewshot_mode sft is set.

Every fine-tune restarts from the original base model and trains only on whatever is
currently in the buffer (up to buffer_size entries) - never continuing from the previous
round's checkpoint - so rounds don't compound drift into each other. This is a deliberate
simplification for the first version (see module-level discussion with the user): no
anchoring, no LoRA, no KL penalty against the base model - just plain full-parameter SFT,
redone from scratch each time. Overfitting on 5-10 examples x 10 epochs is expected to show
up eventually; it's being left alone for now and revisited later (candidates: LoRA so the
"from scratch" restart is cheap enough to try more often, or anchoring against the base
model's own output distribution).

Each round's checkpoint is a full bf16 copy of the base model (~8GB for Qwen3-4B) - with
disk quota tight (practical ceiling well under the nominal quota - see conversation with the
user), only one checkpoint is ever kept on local disk, overwritten every round rather than
accumulating one directory per round. Each round trains into a staging directory and only
once that's confirmed complete does it replace the previous checkpoint (so a training crash
never destroys the last good one; see _run_round) - the previous and new checkpoint briefly
coexist on disk only for the few seconds around that swap, not for the training run's whole
duration (TrainingArguments(save_strategy="no") in online_sft_train.py means nothing is
written until the final save, right at the end). If --online_sft_hf_repo is set, each
successfully-swapped-in checkpoint is also pushed there as a new commit - HF's git-backed
storage keeps every round's full history there even though local disk only ever holds the
latest one.

Runs serially on the single GPU the whole job already has: fine-tuning needs exclusive GPU
access, so each trigger stops the live vLLM server, trains (as a subprocess - see
online_sft_train.py), starts a new vLLM server pointed at the fresh checkpoint, and only then
lets the GA loop's next episode proceed. The pause is synchronous from the caller's point of
view, the same shape as OnlineFewShotBuffer.record_episode()'s own backpressure block.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi


def _poll_models_endpoint(base_url, timeout, want_up, log):
    """Poll base_url + '/v1/models' until it responds (want_up=True) or stops responding
    (want_up=False), or raise TimeoutError. Logs its own progress so a slow cold start -
    expected to take well under a minute for a 4B model, but not guaranteed - doesn't look
    like a hang.

    want_up=False requires 3 *consecutive* failed requests, spaced a couple seconds apart,
    not just one - a single dropped/slow request under heavy load (e.g. the dying server
    itself still mid-request while its process is being torn down) is not reliable evidence
    the server has actually stopped, and treating it as such is exactly what let training
    start against a GPU the old server's engine process hadn't actually released yet (see
    _stop_current_server's nvidia-smi check, added for the same reason after that happened
    live)."""
    deadline = time.time() + timeout
    # base_url already carries the "/v1" prefix (it's "http://localhost:{port}/v1", matching
    # the OpenAI-client base_url convention used throughout this codebase - e.g. --llm_base_url)
    # - appending another "/v1/models" here produced "/v1/v1/models", a 404 that never
    # resolves. Confirmed live: the server was actually up and serving the whole time.
    url = base_url.rstrip("/") + "/models"
    consecutive_down = 0
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            up = True
        except Exception:
            up = False
        if want_up:
            if up:
                return
        else:
            consecutive_down = consecutive_down + 1 if not up else 0
            if consecutive_down >= 3:
                return
        time.sleep(3)
    raise TimeoutError(f"server did not become {'ready' if want_up else 'unreachable'} "
                        f"at {url} within {timeout:.0f}s")


def _gpu_memory_used_mib(gpu_index, log):
    """Best-effort nvidia-smi query; returns None (never raises) if it can't be read, so a
    caller treats "unknown" the same as "can't confirm it's free" rather than crashing on
    something that isn't this feature's core job."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception as e:
        log(f"  [online-sft] nvidia-smi query failed: {type(e).__name__}: {e}", flush=True)
        return None


def _wait_for_gpu_memory_free(gpu_index, threshold_mib, timeout, log):
    """Block until the GPU's used memory drops under threshold_mib, or raise TimeoutError.
    Added after a live failure where the old vLLM server's API process died (so /v1/models
    correctly stopped answering) but its separate EngineCore worker process - vLLM's own
    multiprocess executor spawns the actual model/KV-cache holder as a child, and a plain
    SIGTERM to the top-level process does not reliably cascade to it - kept running and
    holding ~38GB, so the very next fine-tuning round's model load OOM'd immediately. The
    process-group kill in _stop_current_server should prevent that outright now, but this is
    a second, independent check before training is allowed to start: unlike the HTTP
    endpoint, it looks at the actual resource being contended for."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = _gpu_memory_used_mib(gpu_index, log)
        if used is not None and used < threshold_mib:
            return
        time.sleep(3)
    raise TimeoutError(f"GPU {gpu_index} still has >= {threshold_mib}MiB used after "
                        f"{timeout:.0f}s - a previous server's process likely didn't fully exit")


class OnlineSFTManager:
    def __init__(self, buffer, agent, base_model, port, vllm_pidfile, output_dir,
                 trigger_every=5, epochs=10, learning_rate=1e-5,
                 vllm_bin="/u/jgu3/anaconda3/envs/molleo-vllm/bin/vllm",
                 vllm_extra_args=None, tensor_parallel_size=1,
                 server_ready_timeout=600.0, hf_repo=None, hf_private=False,
                 gpu_index=0, served_model=None, round_offset=0, log=print):
        # base_model: what every fine-tuning round restarts from (never updated after
        # construction - see _run_round's module-docstring note on "always restart from
        # base"). served_model: what's ACTUALLY being served right now, which on a resumed
        # chain link (see _run_qwen3_4b_vllm_onlinesft_chain.sbatch) is a previous link's
        # checkpoint path, not base_model - conflating the two would make every future round
        # silently continue fine-tuning from that checkpoint instead of the true base.
        self.buffer = buffer
        self.agent = agent
        self.base_model = base_model
        self.port = port
        self.base_url = f"http://localhost:{port}/v1"
        self.vllm_pidfile = vllm_pidfile
        self.output_dir = os.path.join(output_dir, "online_sft")
        self.trigger_every = trigger_every
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.vllm_bin = vllm_bin
        self.vllm_extra_args = list(vllm_extra_args or [])
        self.tensor_parallel_size = tensor_parallel_size
        self.server_ready_timeout = server_ready_timeout
        self.hf_repo = hf_repo
        # The single GPU this whole job was allocated (--gpus-per-node=1 in every sbatch
        # script that uses online_fewshot_mode=sft so far) - used only for the post-stop
        # "is it actually free yet" check in _stop_current_server.
        self._gpu_index = gpu_index
        self.log = log

        if self.hf_repo:
            self._hf_api = HfApi()
            try:
                self._hf_api.create_repo(repo_id=self.hf_repo, private=hf_private, exist_ok=True)
                self.log(f"  [online-sft] will push each round's checkpoint to "
                         f"https://huggingface.co/{self.hf_repo}", flush=True)
            except Exception as e:
                self.log(f"  [online-sft] could not create/verify HF repo {self.hf_repo}: "
                         f"{type(e).__name__}: {e} - pushes will be attempted anyway and "
                         f"logged if they fail", flush=True)

        # Qwen's chat template needs vLLM's Hermes tool-call parser turned on explicitly
        # (see serve_local_vllm.sh) - that script sniffs the model id's "Qwen/" prefix to
        # decide this, which only matches the *original* HF hub id, never a local
        # fine-tuned checkpoint directory. Decided once, from the original base model id,
        # and reused verbatim for every later restart so a checkpoint path never silently
        # loses the parser flags just because it no longer looks like "Qwen/...".
        self._family_args = (["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
                              if base_model.startswith("Qwen/") else [])

        self._server_proc = None  # Popen for a server this manager itself started, if any
        self._current_model_path = served_model or base_model  # whatever is actually serving now
        self.round = round_offset
        # Baseline for the trigger check below, fixed at construction (not lazily bootstrapped
        # from whatever buffer.n_entries_accepted_total happens to be on the first call) so a
        # buffer that already holds entries before online_sft is wired in - e.g. one resumed
        # via --online_fewshot_load_buffer - still counts every one of them toward the first
        # trigger, rather than silently treating them as already "spent".
        self._last_trigger_count = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def maybe_finetune(self):
        """Call once per episode, after buffer.record_episode(). A no-op unless at least
        trigger_every new entries have been accepted since the last round."""
        total_now = self.buffer.n_entries_accepted_total
        if total_now - self._last_trigger_count < self.trigger_every:
            return
        # Snapshot the trigger baseline now, before the slow work below - any entries a
        # background beam-search cycle accepts while this round is running correctly carry
        # over toward the *next* trigger rather than being silently absorbed by this one.
        self._last_trigger_count = total_now

        entries = self.buffer.export_entries()
        if not entries:
            self.log("  [online-sft] trigger fired but the buffer is empty, skipping this round", flush=True)
            return

        self.round += 1
        self.log(f"  [online-sft] round {self.round}: fine-tuning on {len(entries)} buffer "
                 f"entries ({total_now} accepted so far)...", flush=True)
        t0 = time.time()
        try:
            checkpoint_dir = self._run_round(entries)
        except Exception as e:
            self.log(f"  [online-sft] round {self.round} failed: {type(e).__name__}: {e}", flush=True)
            # _run_round stops the currently-served model before training, so a failure here
            # (training itself, or starting the new server) can leave nothing being served at
            # all - try to bring self._current_model_path back up rather than stranding every
            # later episode with no server to call.
            try:
                _poll_models_endpoint(self.base_url, timeout=1, want_up=True, log=self.log)
            except Exception:
                self.log(f"  [online-sft] no server currently answering - restarting "
                         f"{self._current_model_path} to recover", flush=True)
                # Caught, not left to propagate: maybe_finetune is called from the main GA
                # loop (main/toolmol/run.py's _optimize()) with nothing above it to catch an
                # exception - letting a failed recovery attempt raise here would crash the
                # entire run over a fine-tuning round, which is exactly the failure mode this
                # block exists to avoid in the first place.
                try:
                    self._start_server(self._current_model_path)
                except Exception as e2:
                    self.log(f"  [online-sft] recovery restart of {self._current_model_path} "
                             f"also failed: {type(e2).__name__}: {e2} - no server is currently "
                             f"running; every later episode will fail its LLM calls until this "
                             f"is fixed", flush=True)
            return
        self._current_model_path = checkpoint_dir
        self.log(f"  [online-sft] round {self.round} done in {time.time() - t0:.0f}s, "
                 f"now serving {checkpoint_dir}", flush=True)

    def _run_round(self, entries):
        # Small text files (the messages, not the model) - kept one per round since they cost
        # nothing, unlike the checkpoint itself below.
        train_file = os.path.join(self.output_dir, f"train_entries_round_{self.round}.json")
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False,
                      default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

        staging_dir = os.path.join(self.output_dir, "checkpoint_staging")
        checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)

        self._stop_current_server()
        # Writes nothing until trainer.save_model()/tokenizer.save_pretrained() at the very
        # end (save_strategy="no") - so a crash during training leaves staging_dir empty or
        # absent, and checkpoint_dir (the last good round) untouched, below.
        self._train(train_file, staging_dir)

        # Only replace the previous checkpoint once the new one is fully written - keeps at
        # most one complete checkpoint on disk at a time (per the user's disk-quota ask),
        # with the old and new briefly coexisting only for this swap, not for training's
        # whole duration.
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.rename(staging_dir, checkpoint_dir)

        self._start_server(checkpoint_dir)
        self._push_to_hf(checkpoint_dir)

        self.agent.model = checkpoint_dir
        # The buffer's own post-rationalization calls (_build_example) go through this same
        # served endpoint and need to ask for the model it's actually serving now, same as
        # the live agent - see OnlineFewShotBuffer.update_llm_model's docstring for the live
        # failure this fixes (every post-round-1 post-rationalization call 404'ing).
        self.buffer.update_llm_model(checkpoint_dir)
        return checkpoint_dir

    def _push_to_hf(self, checkpoint_dir):
        if not self.hf_repo:
            return
        try:
            self._hf_api.upload_folder(folder_path=checkpoint_dir, repo_id=self.hf_repo,
                                        commit_message=f"online-sft round {self.round}")
            self.log(f"  [online-sft] pushed round {self.round} checkpoint to "
                     f"https://huggingface.co/{self.hf_repo}", flush=True)
        except Exception as e:
            self.log(f"  [online-sft] push to {self.hf_repo} failed: {type(e).__name__}: {e} "
                     f"- continuing without it, the local checkpoint is unaffected", flush=True)

    def _stop_current_server(self):
        """Stop whatever's on self.vllm_pidfile and confirm the GPU is actually free, not
        just that the HTTP endpoint went quiet.

        Both the initial server (started by the launching sbatch script under setsid - see
        that script's comments) and every server this manager itself starts (start_new_session
        in _start_server) are their own process-group leader with pgid == pid, so killpg here
        reaches every process in that group - the API-server process AND vLLM's own separate
        EngineCore worker process, which a plain kill(pid) does not reliably reach (confirmed
        live: the API server died and /v1/models correctly stopped answering, but its
        EngineCore child kept running and holding ~38GB, so the very next round's model load
        OOM'd on a GPU that looked - by the HTTP check alone - already free)."""
        pid = self._read_pidfile()
        if pid is None:
            self.log("  [online-sft] no server PID on file, assuming nothing to stop", flush=True)
        else:
            self.log(f"  [online-sft] stopping server process group {pid}...", flush=True)
            self._killpg_and_wait(pid, signal.SIGTERM, wait=30)
            if self._pgid_alive(pid):
                self.log(f"  [online-sft] process group {pid} still alive after SIGTERM, "
                         f"escalating to SIGKILL", flush=True)
                self._killpg_and_wait(pid, signal.SIGKILL, wait=30)
                if self._pgid_alive(pid):
                    self.log(f"  [online-sft] process group {pid} still alive after SIGKILL - "
                             f"proceeding anyway", flush=True)
        # Reap this process if it's a direct child of ours (a server this manager itself
        # started in a previous round, via _start_server - not the first round's, which is
        # the launching sbatch script's child, not Python's). Harmless either way - a
        # terminated-but-unreaped process is a zombie, costing a process-table slot and
        # nothing else - but there's no reason to let one accumulate per round over a long run.
        if self._server_proc is not None:
            try:
                self._server_proc.wait(timeout=5)
            except Exception:
                pass
        _poll_models_endpoint(self.base_url, timeout=60, want_up=False, log=self.log)
        _wait_for_gpu_memory_free(self._gpu_index, threshold_mib=2000, timeout=120, log=self.log)

    def _pgid_alive(self, pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, just not signalable by us - still "alive" for our purposes

    def _killpg_and_wait(self, pgid, sig, wait):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline and self._pgid_alive(pgid):
            time.sleep(2)

    def _train(self, train_file, checkpoint_dir):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_sft_train.py")
        cmd = [sys.executable, script,
               "--base_model", self.base_model,
               "--train_file", train_file,
               "--output_dir", checkpoint_dir,
               "--epochs", str(self.epochs),
               "--learning_rate", str(self.learning_rate)]
        self.log(f"  [online-sft] training: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    def _start_server(self, model_path):
        cmd = [self.vllm_bin, "serve", model_path,
               "--port", str(self.port),
               "--tensor-parallel-size", str(self.tensor_parallel_size),
               *self._family_args, *self.vllm_extra_args]
        # self.output_dir (created in __init__), never a path derived from model_path - the
        # recovery call in maybe_finetune's except-block can pass the *original* base_model
        # id (e.g. "Qwen/Qwen3-4B-Instruct-2507") as model_path, whose dirname ("Qwen") isn't
        # a real directory at all.
        log_path = os.path.join(self.output_dir, f"vllm_server_round_{self.round}.log")
        # vllm_bin lives in a different conda env than this process (molleo-vllm vs. whatever
        # launched run.py) - its LD_LIBRARY_PATH/PATH may no longer be the ones in os.environ
        # by now (conda activate/deactivate can rewrite them switching envs), so both are set
        # explicitly here from vllm_bin's own env root rather than trusting inherited state.
        # PATH matters even though vllm_bin is given by full path: vLLM's own JIT compilation
        # (flashinfer's sampling kernel) shells out to `ninja` by bare name at serve time, not
        # just at import time - confirmed live, this failed with FileNotFoundError: 'ninja'
        # on a restart launched from this process's (molleo env's) PATH, which doesn't carry
        # molleo-vllm's bin/ where ninja actually lives.
        env = dict(os.environ)
        env_root = os.path.dirname(os.path.dirname(os.path.abspath(self.vllm_bin)))
        env["LD_LIBRARY_PATH"] = env_root + "/lib:" + env.get("LD_LIBRARY_PATH", "")
        env["PATH"] = env_root + "/bin:" + env.get("PATH", "")
        self.log(f"  [online-sft] starting server: {' '.join(cmd)} (log: {log_path})", flush=True)
        log_f = open(log_path, "w")
        # start_new_session=True makes this process its own session/process-group leader
        # (pgid == pid), same as the setsid the launching sbatch script wraps the *first*
        # server in - so _stop_current_server's killpg(pid, ...) reaches vLLM's own
        # EngineCore worker child too when this server's turn comes to be stopped, not just
        # this top-level process.
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        self._server_proc = proc
        self._write_pidfile(proc.pid)
        try:
            _poll_models_endpoint(self.base_url, timeout=self.server_ready_timeout,
                                   want_up=True, log=self.log)
        except TimeoutError:
            if proc.poll() is not None:
                log_f.close()
                with open(log_path) as f:
                    tail = f.read()[-4000:]
                raise RuntimeError(f"vllm server exited early (code {proc.returncode}); "
                                    f"log tail:\n{tail}")
            raise

    def _read_pidfile(self):
        try:
            with open(self.vllm_pidfile) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def _write_pidfile(self, pid):
        with open(self.vllm_pidfile, "w") as f:
            f.write(str(pid))

    def shutdown(self):
        """Best-effort cleanup of a server this manager itself started (not the original one
        the sbatch script launched, which it already stopped before its first restart) -
        called from the same place GB_GA_Optimizer shuts down online_buffer. The job's own
        cgroup teardown would eventually reap this anyway, but doing it here (via the same
        process-group kill _stop_current_server uses, not a plain terminate() of just the
        top-level process) avoids leaving an orphaned EngineCore worker holding the GPU for
        whatever runs next on this allocation."""
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._killpg_and_wait(self._server_proc.pid, signal.SIGTERM, wait=10)
