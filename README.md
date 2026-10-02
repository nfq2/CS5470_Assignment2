# CS5470 - HW2: Scheduling prompts
**Due: 10/16/2026 11:59pm**

## 1. Overview

This homework focuses on resource scheduling for LLM serving on GPUs with limited memory. You will setup a 34B model on 2 A100 GPUs and measure how much memory the weights occupy and how much memory remains for the KV cache.

The first step is to benchmark and plot how a burst of requests affects the perceived responsiveness of the LLM for each prompt's user. After this, you will implement a series of changes to vLLM's built-in scheduler class to improve responsiveness of the system.

You might find the `Scheduling` section in this [blog](https://medium.com/@crclq2018/explaining-the-source-code-behind-the-vllm-fast-inference-engine-91429f54d1f7) helpful.

## 2. Prerequisites
We will install all the required packages under `/pscratch` for faster compilation, including the vLLM codebase and Conda environment. We suggest you use VSCode SSH extension to work on Perlmutter.

### Apply for model access
Apply for model access on Huggingface
https://huggingface.co/meta-llama/CodeLlama-34b-hf

### Create Conda Environment
```bash
module load conda
conda create -y --prefix $PSCRATCH/sysml-hw2 python=3.10.12
```

### Activate Conda Environment
```bash
conda activate $PSCRATCH/sysml-hw2
conda install -y -c conda-forge gxx_linux-64=13
# ignore packages installed in ~/.local (reactivate to apply)
conda env config vars set PYTHONNOUSERSITE=1 HF_HOME=$PSCRATCH/huggingface
conda deactivate && conda activate $PSCRATCH/sysml-hw2
```

### Clone vLLM Repository
```bash
cd $PSCRATCH
git clone https://github.com/vllm-project/vllm.git
cd vllm
git reset --hard 5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
# We are using v0 version 
```

### Install PyTorch
```bash
python3 -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 -v
```

### Build vLLM
```bash
python3 use_existing_torch.py
export CMAKE_VERBOSE_MAKEFILE=1
pip install setuptools-scm wheel build setuptools regex
conda install -y -c conda-forge cmake=4.4.3
```

```bash
salloc ... # get a GPU node and work from there

module load conda
module unload cudatoolkit
module load cudatoolkit/12.4
conda activate $PSCRATCH/sysml-hw2
cd $PSCRATCH/vllm
# use the conda compiler (GCC 13)
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
# this step will take roughly 20 minutes
MAX_JOBS=32 python3 -m pip install -e . --no-build-isolation -v 
pip install "transformers<4.54.0" 
```

### Install CodeLLama-34b
```bash
hf auth login
huggingface-cli download meta-llama/CodeLlama-34b-hf
```

## 3. Homework Tasks

This homework is designed to understand how serving engines schedule prompts on the GPU and how to make serving more responsive.

Note that this homework requires allocating a GPU node with **two A100 40GB** GPUs on Perlmutter:
```bash
salloc -N 1 -C "gpu&hbm40g" --gpus-per-node 2 -q interactive ...
```

### 3.1 Understanding why the TTFTs are high

Within the homework folder, we have provided `server.sh`, a script that starts a vLLM instance running `codeLlama-34B` on GPU 0 and 1 of the server and exposes an OpenAI compatible HTTP server.

#### Task 1: Experiment on generating a burst of requests

(a) Start the codeLlama-34B model on 2 GPUs using the command provided in server.sh. Note that we store the server log in `vllm_server.log`
> Warning: Logs are **appended** to `vllm_server.log`. Delete it before you start the server to ensure your server log includes only that run.

(b) Execute the benchmark script with `client.sh`. It sends a burst of 50 requests (3 requests per second on average). Use two shells on the same GPU node, one for the server and one for the client. Save the client output and the server log in `results_3.1/`.

Shell 1 (server):
```bash
rm -f vllm_server.log
bash server.sh
```

Shell 2 (client), once the server is ready (`curl localhost:8000/health` succeeds):
```bash
mkdir -p results_3.1
bash client.sh | tee results_3.1/client.txt
```

When the client finishes, stop the server with Ctrl-C in shell 1, then move the log:
```bash
mv vllm_server.log results_3.1/
```

(c) Store the TTFT (Time To First Token) and TPOT (Time Per Output Token) of each prompt from the output and plot the TTFT and TPOT over prompt index in the **issuing order** (you don't need to sort the output here, unlike hw1).  Notice how the TTFTs of requests later in the queue have higher TTFTs.

(d) Plot the GPU KV cache usage over time using the server log.

#### Task 2: Answer the following MCQs

> For each question in this homework, give the answer **and the evidence** listed under the question, measured from **your own** results. Fill both in `answers.txt` (template provided in the homework folder). We check the evidence against the logs you submit. An answer without matching evidence gets no point. "Time" means seconds since the first `Running: ...` stats line of the run (the x-axis of your log plots). "Request index" is the x-axis of your per-request plots (issuing order starting at 0).

1. What is the default scheduling policy of vLLM?
   (Hint: Look at `$PSCRATCH/vllm/vllm/core/scheduler.py`, use the version there)
   
   (a) LCFS (Last Come First Serve)
   
   (b) SJF (Shortest Job First)
   
   (c) FCFS (First Come First Serve)
   
   (d) LJF (Longest Job First)

   Evidence (`results_3.1/client.txt`): TTFT of request index 10 and 40.

2. What is preventing vLLM from batching all the incoming prompts on the GPU?
   
   (a) GPU compute capacity
   
   (b) GPU memory bandwidth
   
   (c) GPU power supply
   
   (d) GPU memory capacity

   Evidence (`results_3.1/vllm_server.log`):
   - Number of GPU KV cache blocks (the `# cuda blocks` line) and how many tokens they hold (16 tokens per block).
   - KV cache size of one token on one GPU, with your calculation (CodeLlama-34B: 48 layers, 8 KV heads, head dimension 128, fp16, split across 2 GPUs), and the resulting KV cache size per GPU in GiB.
   - Peak GPU KV cache usage (%), and the `Running` and `Pending` counts on the first line where the peak appears.

### 3.2 Reducing the TTFT variance during bursts

You will now introduce changes to vLLM's scheduler to implement preemptive scheduling. This scheduling policy mimics Linux's Completely Fair Scheduler (CFS), which relies on preemption. Existing policy admits a certain number of prompts on the GPU, waits for them to complete generating all the tokens and then admits the next batch of prompts. In preemptive scheduling, the policy admits a certain number of prompts, generates a fixed number of tokens for each prompt, preempts the prompts to CPU memory and then loads and executes the prompts waiting in the CPU memory on GPU memory. The policy repeats this in a loop so that all the prompts experience fair progress.

#### Primer on vLLM's scheduler

vLLM implements a scheduler which admits new prompts into the GPU only when there is spare memory capacity. The main entry point to vLLM's scheduler is the function `_schedule` in `scheduler.py` file which calls the `_schedule_default` function which implements the default policy. In this assignment, we will add another function `_schedule_cfs` and change the scheduler's policy to a preemptive policy. The goal of the preemptive scheduler is to preempt lower priority prompts to execute prompts with a higher priority -- the `_get_priority` function implements how the priority is calculated for each prompt. 

vLLM also has a function called `_schedule_priority_preemption` which schedules prompts with higher priority by preempting prompts on the GPU with a lower priority, the default behavior of the function is to preempt the prompts back to the waiting queue and recompute it from the prefill stage again later. 

To implement fair scheduling, we need to (1) update the `_get_priority` function, and (2) trigger preemption based on priority at a reasonable frequency, and (3) not cause too much KV cache re-computation overhead after preemption (the default `_schedule_priority_preemption` will). For the latter, you are going to make edits to `_schedule_priority_preemption`, scheduling the right set of prompts across the waiting prompts (newly arrived), swapped prompts and currently running prompts on the GPU.

Note that `_schedule_running` will also generate preempted sequences, but that is triggered by not having enough budget for current running sequences (the KV cache keeps growing for the running sequences). We don't need to care about it.

> Note: For the following tasks, edit the provided `scheduler.py` file and copy-paste it to vLLM codebase (`$PSCRATCH/vllm/vllm/core/scheduler.py`). You do not need to recompile because the build is editable.

#### Task 1: Implement the CFS scheduler

1. **Interval.** Run the CFS policy (`_schedule_cfs`) once every `CFS_INTERVAL` scheduling steps and the default policy (`_schedule_default`) on the other steps. The provided `scheduler.py` already reads the environment variable `CFS_INTERVAL` and declares `num_iters` for counting steps. With `CFS_INTERVAL=0`, the scheduler must behave exactly like the original (FCFS).

2. **Priority.** A request that has generated fewer tokens has higher priority (`_get_priority`).

3. **Preemption.** In a CFS step, higher-priority requests, including newly arrived ones in the waiting queue, must be able to take GPU memory from lower-priority running requests (`_schedule_priority_preemption`).

4. **Swap out.** Preempted requests are swapped out to CPU memory and later swapped back in, instead of returning to the waiting queue and being recomputed from the prefill (`server.sh` runs vLLM with `--preemption-mode swap`).

   (Hint) Preempting only records which blocks to copy. Make sure that list ends up in the `blocks_to_swap_out` field of the `SchedulerOutputs` returned by `_schedule_cfs`.

> ⚠️ Note that there could be many ways to implement fair scheduling. Your implementation gets full credit if the P99 TTFT at interval 10 is lower than 8000ms, and partial credit decreasing with every second above 8000ms. We measure it with `sweep.sh` on a 2x A100 40GB node, taking the median of 3 runs.

#### Task 2: Sweep the CFS interval and measure the metrics

We provide `sweep.sh`, which runs the benchmark for several CFS intervals. For each interval, it starts the server with `CFS_INTERVAL` set, runs `client.sh`, stops the server, and saves the client output and the server log.

> Stop any server you started in 3.1 before running `sweep.sh` and do not run `server.sh` while it runs. `sweep.sh` starts and stops its own server for every interval.

```bash
bash sweep.sh
```
Results are written to `results_3.2/`:
- `p<interval>_r1.txt`: client output (TTFT and TPOT for each request)
- `server_p<interval>.log`: server log for the interval

Interval 0 is FCFS. As a sanity check, its TTFTs should be close to your results from 3.1.

Write `analyze.py`, which takes the two results directories as arguments (`python3 analyze.py results_3.1 results_3.2`). It writes the 3 plots from 3.1 Task 1 (`ttft.png`, `tpot.png`, `gpu_kv.png`) into `results_3.1/`, and the following 6 plots into `results_3.2/`.

FCFS (interval 0) vs. CFS (interval 10) where x-axis is request index in issuing order and y-axis is:
1. TTFT of each request: `ttft.png`
2. TPOT of each request: `tpot.png`

FCFS (interval 0) vs. CFS (interval 10) where x-axis is time (seconds) since the first logged line of each run and y-axis is:

3. GPU KV cache usage (%): `gpu_kv.png`
4. CPU KV cache usage (%): `cpu_kv.png`
5. Number of waiting requests (`Pending: N reqs` in the server log): `waiting.png`

All intervals:

6. P99 TTFT (y-axis) against P99 TPOT (x-axis) with one labeled point per interval (FCFS, 5, 10, 20, 40): `pareto.png`

If implemented correctly, the TTFTs should reduce. The TPOT will become worse, which is expected due to our scheduling policy.

#### Task 3: Answer the following MCQs

1. How does preemptive scheduling (CFS) reduce TTFT of all the prompts?
   
   (a) By prioritizing existing prompts over new prompts.
   
   (b) By prioritizing prompts with the least number of tokens.
   
   (c) No prioritization is involved.
   
   (d) By skipping some tokens.

   Evidence (`server_p0.log`, `server_p10.log`): how long requests kept waiting in each run, i.e. the seconds from the first stats line with `Pending` > 0 to the first line after it with `Pending` = 0.

2. What does preemptive scheduler store and load while swapping prompts between CPU DRAM and GPU?
   
   (a) KV cache of the prompts.
   
   (b) The current layer weights.
   
   (c) LoRA adapters.
   
   (d) Layer activations.

   Evidence:
   - Number of CPU KV cache blocks (the `# CPU blocks` value).
   - In `server_p10.log`: peak CPU KV cache usage (%), peak `Swapped` count, and the KV cache in CPU memory at that peak in GiB per GPU.
   - In `server_p0.log`: peak CPU KV cache usage (%).

3. What is the main tradeoff of preemptive scheduling?
   
   (a) Preemption changes the accuracy of output.
   
   (b) The GPU is idle while it waits for swapping.
   
   (c) Each request generates tokens more slowly (higher TPOT).
   
   (d) Some prompts never get scheduled.

   Evidence: a table with P99 TTFT, P99 TPOT, and output throughput for every interval (FCFS, 5, 10, 20, 40), from the client summaries.

4. As the CFS interval grows from 5 to 40, what happens?

   (a) TTFT and TPOT both decrease.

   (b) TTFT increases and TPOT decreases. Each further TPOT reduction costs more TTFT.

   (c) TTFT decreases and TPOT increases.

   (d) Neither changes noticeably.

   Evidence: from your table, the change in P99 TTFT (s) and P99 TPOT (ms) from interval 5 to 10 and from 20 to 40, and the TPOT saved (ms) per second of added TTFT for each.

## 4. Deliverables

### 1. Report (PDF format)
- All requested plots (3 plots from 3.1 Task 1 and 6 plots from 3.2 Task 2)
- Answer to the MCQs using the format – section, MCQ #, answer.

### 2. MCQ answers
- `answers.txt`: MCQ answers and evidence in the provided template.

### 3. Code
- `scheduler.py` with the CFS scheduler.
- `analyze.py`

### 4. Data files with benchmark results
- `results_3.1/`: the client output and the server log of your FCFS run from 3.1.
- `results_3.2/`: the output of `sweep.sh`.

### Submission Requirements
Your submission will be one archive named `<netid>_a2.zip` in the following format:
```text
<netid>_a2.zip
└── <netid>_a2/
    ├── report.pdf
    ├── answers.txt
    ├── scheduler.py
    ├── analyze.py
    ├── results_3.1/
    │   ├── client.txt          # output of client.sh
    │   └── vllm_server.log     # server log of the same run
    └── results_3.2/            # output of sweep.sh
        ├── env.txt
        ├── p0_r1.txt
        ├── p5_r1.txt
        ├── p10_r1.txt
        ├── p20_r1.txt
        ├── p40_r1.txt
        ├── server_p0.log
        ├── server_p5.log
        ├── server_p10.log
        ├── server_p20.log
        └── server_p40.log
```
Other files in `results_3.1/` and `results_3.2/` (for example the plots from `analyze.py` or `server_p*.stdout`) may be included. We will run `python3 analyze.py results_3.1 results_3.2` on your submission.

Ensure that your submission matches the format and file names exactly.
