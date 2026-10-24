# Security

hive-ram is research software for joining a few machines you own with a cable. It is **not** built to be exposed to
a network you do not control. Read this before you open a port.

## Threat model

**Assumed trusted:** every machine in the chain, every user and process on those machines (including every WSL
distribution on a Windows node), and the link between them: a point-to-point USB4 / Thunderbolt cable, or an
isolated switch that only those machines are plugged into.

**Not defended against:** anyone who can reach a stage port, the llama.cpp RPC port or an HTTP port. Such a peer can
use your GPUs, read prompts and replies, feed wrong activations into the chain or crash it. The project does nothing
to stop a hostile peer on the same network segment beyond the checks listed below.

| Surface | Default | Authentication | Encryption |
|---|---|---|---|
| Flash-Next tfpp rank links (`tfpp/tfpp/wire.py`) | TCP, binds `TFPP_BIND` / `--bind`, default `0.0.0.0` | **Off by default.** Optional `TFPP_SECRET`: both ends prove the shared secret (HMAC-SHA256 over both nonces) when the link is set up; later frames carry no MAC | None |
| tfpp-glm and tfpp-ds41 stage links | TCP, `LISTEN=` in `node.env` (the examples use `0.0.0.0:<port>` inside WSL, behind the relay) | **None** | None |
| tfpp-glm HTTP API (OpenAI-style, leader only) | `serve.py` defaults to `127.0.0.1`; `scripts/glm/cluster.env` sets `HTTP_HOST=0.0.0.0` | **None** | None (plain HTTP) |
| tfpp-ds41 HTTP API (leader only) | `scripts/ds41/cluster.env` sets `HTTP_HOST=0.0.0.0` | Optional bearer key from the fork's server: `TF_API_KEY_FILE` in the leader's `node.env` (every route but `/health`) | None (plain HTTP) |
| llama.cpp `rpc-server` (baseline) | whatever you start it with | **None**: a client can read and write the server's memory buffers | None |
| `linktest.py` | binds `0.0.0.0` unless given an address | None | None |
| Windows relay (`relay_user.ps1`, `start_n1x_relay.ps1`) | listens on the link address, forwards to WSL | None (forwards bytes) | None |

What the code does check, which limits accidents but is **not** a security boundary:

- Frames are a length, a size-limited JSON header and raw tensor bytes. Nothing is unpickled or evaluated. Each
  frame's parts (dtype, shape, byte count) are checked against what the receiver expects before the payload is used;
  the Flash-Next pipeline also checks token ids. A peer that does not speak the link handshake (a port scanner, an
  SSH or TLS client) is dropped and the stage keeps listening.
- At start the stages exchange a HELLO and compare their layer ranges, settings and checkpoint fingerprints (tfpp-glm
  and tfpp-ds41 also compare a digest of the numerics-relevant environment and the engine tree digest); any
  difference stops the chain. These are consistency checks against mistakes, not authentication.
- tfpp-glm and tfpp-ds41 refuse to start on an engine tree whose digest differs from the pinned one
  (`python -m tfpp.glm.pins`, `python -m tfpp.ds41.pins`), so a modified TensorFold tree is not run by accident.
  Flash-Next tfpp has no such check (see [`THIRD_PARTY.md`](THIRD_PARTY.md) to compare the digest yourself).
- tfpp compiles patched copies of functions read from the installed TensorFold tree (see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Whoever can write to that tree or to your checkout can run
  code in every stage: keep both owned by the user that runs the stages.

## Recommended set-up

1. **Use a dedicated link.** Cable the machines directly (no dock, no shared LAN, never Wi-Fi). Give each cable its
   own small subnet; the scripts use `10.77.0.0/24` (and `10.78.0.0/24` for a second cable).
2. **Firewall to the peer only.** `setup_usb4_link.ps1` sets a static address on the USB4 adapter only and adds
   inbound rules that admit ICMP and the given TCP ports only from the peer's address and only to the link address.
   `undo_usb4_link.ps1` removes them. On Linux nodes, do the same with your firewall.
3. **Bind to the link address** wherever the process holds that address itself: `--bind 10.77.0.x` / `TFPP_BIND` for
   Flash-Next tfpp, `LISTEN=10.77.0.x:<port>` for tfpp-glm and tfpp-ds41, the address argument of `linktest.py`. Under
   WSL2 NAT the process must listen inside WSL and the Windows-side relay holds the link address; then remember that
   the WSL address the relay forwards to is reachable by every process on that Windows host and every WSL
   distribution, whatever the firewall says.
4. **Set `TFPP_SECRET`** (the same value on every node) for Flash-Next tfpp links. It authenticates the link set-up
   only; it does not encrypt.
5. **Keep HTTP local.** Serve the API on `127.0.0.1` (`HTTP_HOST=127.0.0.1`) and reach it through an SSH tunnel or a
   reverse proxy with authentication and TLS. If you must listen on the link address, for tfpp-ds41 set
   `TF_API_KEY_FILE` (a file readable only by you, `chmod 600`); tfpp-glm has no key at all.
6. **Run servers only while you use them.** Start llama.cpp's `rpc-server`, `linktest.py` and the relays for a test
   and stop them afterwards.
7. **Run elevated scripts from a folder only administrators can change.** `setup_usb4_link.ps1`,
   `undo_usb4_link.ps1` and `start_n1x_relay.ps1` need an administrator PowerShell; never run them from a folder
   other users can write to.
8. **Keep secrets out of shared files.** `cluster.env` is identical on every node and its values are digested into
   HELLO; put per-node secrets (`TF_API_KEY_FILE`, `TFPP_SECRET`) in the environment or in `node.env`, which is not
   tracked (`.gitignore`). tfpp-ds41 never prints or digests variables whose names contain `KEY`, `TOKEN`, `SECRET`
   or `PASSWORD`.

## Models and downloads

Nothing downloads automatically: the scripts set `HF_HUB_OFFLINE=1` and load only folders you downloaded. The engines
load safetensors checkpoints. `VERIFY_SHARDS=1` makes a stage check its shards against the Hub's sha256 values
recorded at download time. Third-party engine trees are pinned by digest (see [`THIRD_PARTY.md`](THIRD_PARTY.md)).

## Reporting a vulnerability

Please report security problems privately through GitHub's **"Report a vulnerability"** button on this repository's
Security tab (private vulnerability reporting). If that is not available, open an issue that says only that you have
a security report and asks for a private channel; do not include details or exploit code in public issues.

Include what you ran (variant, commit, command line and `node.env` without secrets; replace host names, user names and
non-link IP addresses in logs, since stages log each node's hostname at HELLO), what an attacker needs (network
position, access), and what happens. We will acknowledge the report, work on a fix in private, and credit you in the
release notes unless you prefer otherwise.

Problems in TensorFold, the MIA recipe patches, the BertholomusAI fork or llama.cpp belong to those projects; report
them upstream (we are happy to help when the issue only shows up in a pipeline split).

## Supported versions

Only the latest commit of the default branch is supported. There are no releases with security backports yet.
