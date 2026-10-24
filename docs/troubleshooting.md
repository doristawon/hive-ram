# Troubleshooting

Lessons learned while bringing up the cluster on two N1X laptops (a 128 GB node and a 64 GB node, Windows 11 with
WSL2 in NAT mode, one USB4 / Thunderbolt cable). Each row gives what you see, why it happens, and what to do.
[hardware-setup.md](hardware-setup.md) explains the setup that these rows refer to. Its
[checklist](hardware-setup.md#9-checklist-before-every-run) catches most of them before a run.

Tags in the Fix column say how far each fix has been proven:

- **(hardware)**: used on the real nodes.
- **(CPU tests)**: in the code and covered by CPU-only tests, but not yet run on hardware.
- **(advice)**: a suggestion we have not tried.

## Contents

- [Link and cabling](#link-and-cabling)
- [Windows networking, relay and firewall](#windows-networking-relay-and-firewall)
- [Processes that die with a remote session](#processes-that-die-with-a-remote-session)
- [WSL memory and disk](#wsl-memory-and-disk)
- [GPU memory and clocks](#gpu-memory-and-clocks)
- [Updates and reboots](#updates-and-reboots)
- [Toolchain](#toolchain)
- [Shell scripts](#shell-scripts)
- [llama.cpp on N1X](#llamacpp-on-n1x)
- [Bit-exact comparisons](#bit-exact-comparisons)

## Link and cabling

| Symptom | Cause | Fix |
|---|---|---|
| **No `USB4(TM) P2P Network Adapter`** appears on one machine. Replugging and other ports change nothing, and no device events are logged. | Host-to-host USB4 networking did not come up during that boot. We first suspected a firmware difference between the two machines, but the link later came up on the same firmware, so that was not the cause. | Plug the cable straight into a USB4 / Thunderbolt port on each machine, with no dock or hub in between. Reboot the machine that shows no adapter **(hardware)**. To bring up the software meanwhile, wired Ethernet through a switch should work at 1 Gbit/s **(advice)**. |
| The adapter is `Up` but holds only a 169.254.x.x address, and the peer does not answer. | `setup_usb4_link.ps1` has not run on both ends. | Run it on both machines (hardware-setup section 3.2). |
| `cannot tell which USB4 adapter is this link` | Several USB4 adapters are up, and none of them holds the address yet. | Run it again with `-InterfaceAlias` or `-InterfaceIndex`, taken from the list the script prints **(CPU tests)**. |
| `... overlaps ... give this link its own subnet` | Another interface already holds an address in that subnet: a second cable on this node, or the same address left on the adapter of the port you used before. | Give each cable its own subnet (10.77.0.x, 10.78.0.x, ...). For a leftover address, run `undo_usb4_link.ps1 -Address <that address>`, then set up again. |
| The link stopped working after the cable moved to another port. | Each USB4 port can show up as its own adapter. The static address, and the local address of the firewall rules, stayed with the old one. | Move the cable back, or undo that address and run setup again for the new adapter. Use the same port every time. |
| `peer ... does not answer ping yet` | The other end is not set up yet, or the cable is in a different port from the adapter that was configured. | Run setup on the other end. With several USB4 adapters up, pass `-InterfaceAlias`. |
| `LinkSpeed` shows 20 Gbps. | A Thunderbolt 3 cable. | It is fine for this workload: the path into WSL tops out near 5 Gbit/s anyway. A Thunderbolt 4 / USB4 40 Gbit/s cable is recommended, but we have not measured one. |

## Windows networking, relay and firewall

| Symptom | Cause | Fix |
|---|---|---|
| Throughput into WSL is **about 0.4 Gbit/s**, with 20 ms p99 round trips. | Windows' `netsh interface portproxy` (the IP Helper service) is the bottleneck. | Use the userland relay `relay_user.ps1`: 5.2 Gbit/s and 0.49 ms p50 end to end **(hardware)**. |
| The relay will not start: its bind on `10.77.0.1:PORT` fails. | One of three things: a portproxy entry holds the port (from `-PublishPorts` or an older setup); another relay is already running; or no adapter holds that address. | Run `start_n1x_relay.ps1` once as administrator: it deletes the portproxy entries on that address and those ports, then starts the relay. Or list the entries with `netsh interface portproxy show v4tov4`. Stop any other relay, and check `Get-NetIPAddress -IPAddress 10.77.0.1`. |
| Everything worked, then every connection fails after `wsl --shutdown`, a WSL update or a reboot. | WSL's NAT address changed, and `relay_user.ps1` still forwards to the old one. | Restart `relay_user.ps1` after WSL restarts **(hardware)**. `relay_hop.ps1` looks the address up again after a failed connect, unless it was given `-WslIp` **(CPU tests, loopback)**. |
| `no WSL address found`, or a relay started from another Windows account forwards nowhere. | WSL is not running, or the relay's account does not own the distro (WSL VMs are per Windows user). | Start WSL first. Pass `-WslIp` (inside WSL, `ip -4 -o addr show eth0` prints it), or use `relay_hop.ps1 -PortproxyPort N`. |
| The relay logs `forward failed`, and the peer sees its connection closed at once. | Nothing listens on that port inside WSL yet, or the WSL server is bound to 127.0.0.1. | Start the WSL side, bound to `0.0.0.0` or WSL's `eth0` address on the same port. tfpp stages retry the link by themselves until `--connect-timeout`. |
| The peer's connection times out although the relay is running. | The firewall rules admit only the listed ports, only from the peer, and only to this link's address. A new stage port, or a chain that now runs the other way, is not covered. | Run setup again with `-AllowTcpPorts`, listing every port you need: given with `-Node`, it replaces the preset list. Every receiving node needs its own rule and its own relay. In our two-node runs only one node accepted inbound connections, which fixed the chain's direction **(hardware)**. |
| **Firewall scope.** A stage port answers on Wi-Fi or the LAN, or a port scan or health check makes a stage's start fail (exit 75). | A rule without a local and remote address scope, or a server listening on every interface. Older tfpp-glm listeners also took the first connection that arrived as the stage link. | Use the scoped rules from `setup_usb4_link.ps1` (group `hive-ram`: only from the peer, only to the link address). Bind the link address where you can (`--bind` / `TFPP_BIND` for Flash-Next tfpp), and optionally set `TFPP_SECRET` on every node. tfpp-glm's listener now waits for a real LINK handshake and drops anything else **(CPU tests)**. Remember that the WSL address and Windows' 127.0.0.1 are reachable by every local process, whatever the firewall says. |
| A node lost power, and the chain noticed only at the next request. | A half-open TCP connection through the relay, with no keepalive. | `relay_hop.ps1` enables keepalive (60 s idle, 10 s probes). The stage behind it then sees the link close and stops the chain, which is fail-stop by design **(CPU tests; time to detect not measured)**. |
| An upstream stage sat in "handshake" for up to an hour. | An older relay accepted the connection while nothing listened in WSL, and kept the socket open. | The relays now close both sides on a failed forward, and tfpp-glm retries the LINK handshake every second until `--connect-timeout` **(CPU tests, loopback)**. |
| Setup warns that it could not set the network category to Private. | Windows had not created the connection profile yet; the script retries for about 10 s. | Harmless: the firewall rules do not depend on the category. |

## Processes that die with a remote session

| Symptom | Cause | Fix |
|---|---|---|
| **The relay dies when the remote shell that started it disconnects.** | Windows' OpenSSH server kills a session's child processes when the session ends. | Start the relay in a console window on the machine's own desktop, and keep that window open **(hardware)**. If you must start it remotely, launch it outside the SSH session, for example as a scheduled task that runs in your desktop session **(advice)**. |
| A WSL server (`rpc-server`, the `linktest.py` server, a stage) dies when its terminal or remote session closes. | It is a child of that session. | Start stages with `start_stage.sh`, which detaches them with `setsid nohup` and returns once they are up, or with the systemd user units `tfpp-glm@.service` / `tfpp-ds41@.service`. For llama.cpp, start a transient unit, e.g. `sudo systemd-run --unit=<name> --uid=$USER -p MemoryHigh=8G <command>`; it survives the SSH session that started it **(hardware)**. |
| **`pkill -f` kills your own session**, or `pgrep -f` reports a process that is not there. | The command line of the shell running `pkill` contains the pattern too, for example `bash -c "pkill -f linktest"` sent over a remote shell. So the pattern matches the caller. | Anchor the pattern to the program, e.g. `pkill -f '^python3 linktest'` or `pkill -f '^ggml-rpc-server'` **(advice)**. Or use the bracket trick the stage scripts use: `pgrep -f 'tfpp[.]glm[.]serve'` matches the stage but not its own command line **(CPU tests)**. |

## WSL memory and disk

| Symptom | Cause | Fix |
|---|---|---|
| Windows' free memory collapses during or after a big download or copy inside WSL; the WSL VM sits at its memory limit. | The Linux page cache inside WSL counts as Windows committed memory, and WSL returns it to Windows only slowly. | After downloads and copies: `sync && echo 1 \| sudo tee /proc/sys/vm/drop_caches` **(hardware)**. Watch "Committed" in Task Manager. |
| While a stage loads, Windows' commit limit runs out, the pagefile grows to about 50 GB, and **C: drops below 1 GB**. | 77 GB of page cache plus a 57 GiB GPU stage, on one unified-memory machine. | Drop the cache, as above **(hardware)**. tfpp-glm's loader now calls `load.drop_cached()` (`posix_fadvise(DONTNEED)` on the shards, no root) before loading, and reads with O_DIRECT (covered by a CPU test). The pagefile only shrinks after a reboot. Keep C: above about 6 GB. |
| llama.cpp loads fill the page cache. | Buffered reads of the GGUF. | Load with `-lm dio` (direct I/O). Do not give `rpc-server` a cache (`-c`) on a node short of disk or memory **(hardware)**. |
| C:'s free space shrinks while you download into WSL. | The WSL disk file (`ext4.vhdx`) grows once ext4 has no free space (slack) left inside it. | Room for new data ≈ C: free + (vhdx file size - ext4 space used). Measure both before a big download (hardware-setup section 7.1). `tfpp/scripts/ds41/download_list.sh` refuses unless both the WSL disk and C: have 75 GB free (`--vhdx-ok`, `--no-disk-check`) **(CPU tests)**. |
| Deleting files in WSL gives no space back to C:. | The vhdx never shrinks by itself. | Later writes reuse the freed space. To shrink the file, compact it only after `wsl --shutdown` and with a backup **(advice)**. |
| `df -h /` in WSL shows hundreds of GB free while C: is nearly full. | `df` reports the vhdx's maximum size, not the free space on C:. | Use the formula above. |
| Reading weights from `/mnt/c/...` is very slow. | `/mnt/c` goes through WSL's Windows file-sharing layer. | Keep models on WSL's own ext4 disk **(advice; the loaders were only ever run on ext4)**. |
| Shell scripts fail in WSL with `$'\r': command not found`, or `python -m tfpp.ds41.pins --check` refuses the tree. | CRLF line endings from a checkout made on Windows. | Clone inside WSL with git, or set `core.autocrlf false`. For the DeepSeek fork, extract it with `git archive`, as `tfpp/scripts/ds41/toolchain.sh` shows. |
| A stage start warns that the checkpoint's revision is `unknown`, or the stages disagree on it. | The model folder was copied without its hidden `.cache/huggingface` folder. | Copy that folder along with the shards. |

## GPU memory and clocks

| Symptom | Cause | Fix |
|---|---|---|
| **Idle GPU downclock.** A pipeline split is far slower than the sum of its stages. One stage's decode step takes nearly twice as long as when it runs back to back. | After 20-40 ms of idling, the GPU lowers its clocks. A 16-layer step (layers 32–47 + head) took 8.7–9.0 ms back to back and 15–16.5 ms after a gap. A busy CPU does not help (15.9 ms), but one spinning GPU block on a side stream does (9.9 ms). In a pipeline every stage idles while the others work. | KeepWarm (`tfpp/tfpp/warm.py`): while a stage waits on the link, it runs about 1 ms GPU spins on a side stream. `TFPP_WARM_MS` sets the spin length (default 1.0); `0` turns it off. The Flash-Next split went from 24.5-25.1 to 33.0-33.8 tok/s, as fast as one node **(hardware)**. To see the effect on your GPU, run `rank_bench.py ... --modes sleep@0,sleep@30,cpuspin@30,gpuwarm@30`. llama.cpp RPC probably loses the same way; we did not measure it. |
| After a reboot a load fails, or free GPU memory is far lower than expected, and Windows is down to about 2 GB free. | Other GPU programs on the node took the memory. | Stop them before starting a stage **(hardware)**. List their ports in `GPU_GUARD_PORTS` (`node.env`), so that `start_stage.sh` refuses to start beside them unless `ALLOW_GPU_SHARING=1` **(CPU tests)**. |
| A stage runs out of GPU memory in the middle of a request, although its start passed. | Startup measures free memory once (the CAP step). A workload that grew afterwards took the room. In tfpp-ds41, round graphs for long contexts are also captured lazily, outside that check. | Run nothing else on the GPU during a run. Use `ALLOW_GPU_SHARING=1` only knowingly, and give stages that share one GPU a `BUDGET_GIB`. tfpp-ds41: `TFPP_DS41_WARM_ALL_BUCKETS=1` captures every graph at startup **(not run on hardware)**. |
| After an update, Windows sees only half of the RAM. | A firmware update reserved about 32 GB as dedicated GPU memory on the 64 GB node (Windows went from 54.7 GB to 32.2 GB). Windows does not count dedicated GPU memory as RAM. | Nothing is broken: CUDA can still use the carve-out. Measure the stage budget again (`torch.cuda.mem_get_info()` inside WSL) and move the layer cuts if needed. |
| CUDA inside WSL sees less memory than you expected. | Other GPU programs, the carve-out, or possibly WSL's own memory limit. On both our nodes the CUDA total sat just under `.wslconfig`'s `memory=`, and we have **not verified** whether that limit caps it. | Set `memory=` explicitly (WSL's default is half the RAM), run `wsl --shutdown`, and measure again. |

## Updates and reboots

| Symptom | Cause | Fix |
|---|---|---|
| A machine rebooted by itself. Afterwards the relay is gone, WSL has a new address, other GPU programs may be running again, and sometimes there is a new driver or firmware. | The NVIDIA App updated the GPU driver automatically and rebooted, and Windows Update installed a firmware update and rebooted. | During experiments, turn off automatic driver updates in the NVIDIA App and pause Windows Update **(advice)**. After any reboot, go through the [checklist](hardware-setup.md#9-checklist-before-every-run): the static addresses and firewall rules survive; relays, the WSL address and GPU memory do not. |
| Runs that should match across machines differ. | Different GPU driver versions on the nodes may change kernel or cuBLAS results. | Keep the same driver on every node **(advice)**. tfpp-ds41 has a test for exactly this, T-X0, which has not run yet. |

## Toolchain

| Symptom | Cause | Fix |
|---|---|---|
| **`libcudart.so.12` beside CUDA 13.** JIT-built extensions link `libcudart.so.12` next to torch's CUDA 13 runtime (`readelf -d <ext>.so \| grep libcudart`). `start_stage.sh` stops with `extensions linked against a CUDA 12 runtime beside torch's 13`. | pip's CUDA 13 ships `libcudart.so.13` without the `libcudart.so` link that `-lcudart` looks for, and it includes the CCCL headers (`nv/target`, `cuda/std`) only with the `[cccl]` extra. The linker and preprocessor then fall through to Ubuntu's CUDA 12 development packages (`nvidia-cuda-dev`, `libcu++-dev`). | `toolchain.sh` creates the link inside the venv, puts it first on `LIBRARY_PATH`, requires `cuda-toolkit[cccl]==13.0.3`, and builds into a fresh `ext-cu13/` cache. Move the old extensions folder aside and rebuild. After the rebuild, both nodes linked `libcudart.so.13` and the kernel check still passed **(hardware)**. |
| A llama.cpp CUDA build links the system's CUDA 12.4 instead of 13.1. | CMake found the distribution's CUDA first. | Configure with `-DCUDAToolkit_ROOT=/usr/local/cuda-13.1` (your CUDA 13 path) **(hardware)**. |
| torch is not a CUDA 13 build, or a kernel does not run on `sm_121`. | torch was installed from the default package index. | Install from `--index-url https://download.pytorch.org/whl/cu130`. The tfpp-glm preflight checks `torch.version.cuda` and runs a small kernel before loading anything **(CPU tests)**. |
| `Triton X.Y: MIA's CUDA ports need 3.7.x`, or other import errors after installing a second engine. | The engines pin different TensorFold trees and Triton versions. | Use one venv per engine (`~/tf-venv-061`, `~/tf-venv-mia`, `~/tf-venv-dsv41`). The tfpp-glm and tfpp-ds41 toolchains refuse the other engines' venvs. |
| A kernel build fails because `Python.h` is missing. | There is no `python3.14-dev` package on the image. | Extract the Python headers under `~/pyinc`; the env and toolchain scripts put them on the include path. |
| **Python 3.14 dataclasses.** Running a patched copy of a module that defines dataclasses fails inside `dataclasses` while it resolves annotations. | Python 3.14's dataclasses resolve string annotations through `sys.modules[cls.__module__]`. A module made with `types.ModuleType` and `exec` is not in `sys.modules`. | Register the new module in `sys.modules` before `exec`, as `tfpp/tfpp/glm/load.py` (`module_for`) does **(hardware)**. |
| `rpc-server: command not found` with a recent llama.cpp build. | Newer builds name the binary `ggml-rpc-server`. | Use `ggml-rpc-server`. |

## Shell scripts

| Symptom | Cause | Fix |
|---|---|---|
| **`pipefail` with `grep -q`.** A check like `producer \| grep -q PATTERN` fails even though PATTERN is in the output. | `grep -q` exits at its first match. The producer then gets SIGPIPE (in Python, `BrokenPipeError`) and exits non-zero, and `set -o pipefail` fails the whole pipeline. | Write the output to a file first, then `grep` the file. This first showed up as a false failure in a fault-injection test on the hardware; the test scripts now do it this way **(hardware)**. |
| A script under `set -euo pipefail` exits silently with status 1 or 2, for example on a first start or when nothing is left to stop. | A command that fails harmlessly ends the script: `find` on a folder that does not exist yet, or `ls` with no rotated logs. | Tolerate the expected failure (`\|\| true`) **(CPU tests)**. |
| `start_stage.sh` says the stage is "already running" when it is not, or `stop_stage.sh` signals an unrelated process. | A stale pid file after a crash or a WSL restart: the number now belongs to another process. | The scripts now accept a pid only while `/proc/<pid>/cmdline` is that stage's command **(CPU tests)**. |
| A test run's own `CONTEXT` or `HTTP_PORT` is silently replaced. | The script read `cluster.env` after setting its own values. | Read the shared file first, then apply the caller's values. The tfpp-ds41 scripts save and restore them. |
| Setting `SO_SNDBUF` / `SO_RCVBUF` to 8 MiB inside WSL does not give 8 MiB: reading the value back gives 425,984 bytes. | Linux caps an explicit buffer size at twice `net.core.[wr]mem_max`, and an explicit size turns TCP autotuning off, which is expected to cut the rate of large prompt chunks over the link (not measured). | Leave socket buffers to autotuning, as the tfpp-glm and tfpp-ds41 links now do; they log the real sizes at connect **(CPU tests)**. |

## llama.cpp on N1X

| Symptom | Cause | Fix |
|---|---|---|
| The n-gram (PLE) table loads whole into `llama-server`'s memory (RSS about 32 GB) although `--lazy-mode auto` is set. | ggml-cuda treats the N1X as an integrated GPU, and `auto` turns lazy loading off there. With `-lm dio`, the table is then read in full. | Use `--lazy-mode on` **(hardware)**. |
| Decode drops to about 8 tok/s with `-lm mmap`, and 300 tokens read 21 GB from disk. | Without forced lazy mode, llama.cpp marks the whole file `FADV_SEQUENTIAL`. With WSL's `read_ahead_kb` of 8192, each 90-byte table lookup pulls in about 16 MB. | Use `-lm dio --lazy-mode on -ngl all --fit off`: the GPU weights load with direct I/O, and only the table is mapped **(hardware)**. |
| `--fit on` leaves some expert layers on the CPU. | It measured free memory while a previous process still held it. | Use `-ngl all --fit off` **(hardware)**. |
| `rpc-server` fills the WSL disk or the page cache. | Its cache option (`-c`) writes tensor files. | Leave `-c` off on a node short of disk or memory **(hardware)**. |

## Bit-exact comparisons

| Symptom | Cause | Fix |
|---|---|---|
| A layer-by-layer kernel comparison fails on a buffer that no kernel wrote for that window. | Both copies hold uninitialised memory (`torch.empty`). | Compare only the buffers the kernels write in that window **(CPU tests)**. |
| Two cut points give different results under kept-prompt memory pressure. | The kept-prompt byte cap is not cut-invariant: each leader counts only its own layers' bytes, so the two runs evict different states. | For bit-exact tests, keep the byte cap out of play. Under pressure, compare reply tokens only (`compare --tokens-only`) **(hardware)**. |
| A comparison passes although nothing ran. | Two empty traces, or two runs that both stopped mid-request, compared as equal. | `compare` now reports `INCOMPLETE` and fails **(CPU tests)**. |
