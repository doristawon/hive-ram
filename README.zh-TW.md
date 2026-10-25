# hive-ram

**用一條 Thunderbolt／USB4 線把 2–4 台一般電腦接起來，合併它們的記憶體，跑任何一台都放不下的量化 LLM。**

hive-ram 把模型切成幾段**連續的層範圍**（管線平行，pipeline parallelism），每段放在記憶體裝得下它的那台機器上；
機器之間只傳殘差流（residual stream），走線上的一般 TCP。每段的大小可以不一樣：128 GB 的機器放大部分的層，
64 GB 的機器放剩下的。

hive-ram 不是新的推論引擎，而是建立在現有引擎之上：

- **llama.cpp RPC**：本來就能依層切分模型，現在就能用。hive-ram 只提供連線設定和實測數據。
- **tfpp**：我們為 [TensorFold](https://github.com/ashhart/TensorFold)（Python＋CUDA）寫的管線層，每個模型家族一個版本：
  Qwen3.8-Flash-Next（`tfpp`）、GLM-5.3-Flash（`tfpp-glm`）、DeepSeek-V4.1-Flash（`tfpp-ds41`）。

[English README](README.md) · [文件索引](docs/README.md)

> **狀態：研究用程式碼。** 所有東西都是在兩台 NVIDIA N1X 筆電（128 GB 與 64 GB 統一記憶體，Windows 11＋WSL2）
> 用一條 USB4 線串起來開發與量測的。有些部分已在這組硬體上驗證；有些部分只寫好，只做過靜態檢查或 CPU 測試。
> [狀態表](#引擎與驗證狀態)寫明了哪些是哪些，打算花一個週末玩之前請先看。

## 目錄

- [運作方式](#運作方式)
- [為什麼走 USB4 管線而不是張量平行](#為什麼走-usb4-管線而不是張量平行)
- [實測結果](#實測結果)
- [引擎與驗證狀態](#引擎與驗證狀態)
- [需要準備什麼](#需要準備什麼)
- [快速開始](#快速開始)
- [Repo 結構](#repo-結構)
- [限制與注意事項](#限制與注意事項)
- [路線圖](#路線圖)
- [授權](#授權)

## 運作方式

每台機器跑一個 **stage（段）**：一段連續的解碼層。整條鏈上只有一份模型權重，所以各段加起來需要的記憶體，
大約等於一台大機器跑整個模型所需。

```mermaid
flowchart LR
    S0["第 0 段<br/>嵌入層 + 第 [0, a) 層"] -- "殘差列（bf16）" --> S1["第 1 段<br/>第 [a, b) 層"]
    S1 -- "殘差列（bf16）" --> S2["最後一段<br/>第 [b, L) 層 + norm + 輸出頭<br/>取樣、驗證草稿、草稿器"]
    S2 -. "保留幾個 token + 下一批草稿（幾個 id）" .-> S1
    S1 -. "轉送" .-> S0
```

- **一輪（round）**就是沿著鏈走一趟。每一段都對同樣的幾列（待處理的 token 加上草稿 token）跑自己的層；最後一段
  取樣、驗證草稿、提出下一批草稿。它的決定再沿鏈傳回去，所以每一段寫進快取的列完全相同。
- **每一列在線上傳多少**：Qwen3.8-Flash-Next 20 KB（4 × 2,560 個 bf16）；GLM-5.3-Flash 32 KiB（4 × 4,096 個
  bf16），DFlash2 每個 tap 每列另加 8 KiB；DeepSeek-V4.1-Flash 40 KiB（4 × 5,120 個 bf16），切點落在共用 KV
  的層群組內時，另加少量資料（top-k、候選池、壓縮 KV 列）。
- **誰當 leader**：Flash-Next 測試程式由第 0 個 rank 做嵌入並帶動整條鏈。tfpp-glm 和 tfpp-ds41 則由**最後一段**
  當 leader：輸出頭、草稿器、排程和 HTTP API 都在它身上。
- **KeepWarm**：等網路的那一段，會在側邊 stream 上跑一個很小的空轉 kernel（只佔一個 block，每次約 1 ms），
  讓 GPU 保持忙碌。少了它，GPU 每次等待都會降頻，切開後比單機慢約 25%；加上它，切開後和單機一樣快（見下文）。

WSL2 使用預設 NAT 模式時，實際的資料路徑：

```
 節點 A（Windows 11）                                  節點 B（Windows 11）
 WSL2：第 k 段 ──TCP──► USB4 P2P 網卡 10.77.0.1 ═══線═══► 10.77.0.2 ──► 中繼程式（一般使用者）──► WSL2：第 k+1 段
       （從 WSL 往外連不需設定）                          （往 WSL 裡連需要中繼：Windows 內建的
                                                           portproxy 只跑得到 0.42 Gbit/s）
```

## 為什麼走 USB4 管線而不是張量平行

上游 TensorFold 跨機器只支援張量平行（或專家平行，以下統稱 TP）。它自己的文件也寫明：把一個模型的層分散到
幾台較小的機器上需要 pipeline mode，而它沒有做。在筆電對筆電的 USB4 線上，TP 是划不來的選擇：

| | 張量平行（上游跨機器的做法） | 層範圍管線（hive-ram） |
|---|---|---|
| 每產生一個 token 的通訊次數 | 每層 2 次 all-gather：GLM-5.3-Flash 每步約 90 次集體通訊 | 每輪在每個段邊界傳一次 |
| 在我們的線上的代價（經中繼來回約 0.5 ms） | 每步光通訊就要約 22–45 ms（估算）；同樣的集體通訊在 RoCE 網路上約 1 ms | 兩段的鏈每輪約 2.3–2.5 ms，含 Python（實測） |
| 記憶體分配 | 每個 rank 平均分攤，最小那台決定上限。TensorFold 0.6.x 在 CUDA 上跑 GLM 必須剛好 2 個 rank | 任意連續層範圍，依各台的可用記憶體決定 |
| 傳輸方式 | NCCL：隨機 port、節點之間直接互連。WSL 的 NAT 加上單一 port 的中繼承載不了（WSL mirrored 網路未測） | 每一跳一條 TCP 連線，經過一般使用者權限的中繼 |
| 單路速度 | 所有 GPU 同時讀權重 | 同一時間只有一張 GPU 在算：大約一台機器的速度，**不會變快** |

**得到什麼、得不到什麼。** 管線給你的是*容量*：任何一台都放不下的模型，可以用大約「一台機器讀完全部權重」
的速度跑起來。它不會讓單一請求變快。tfpp-glm v2.0（單 GPU 已驗證、尚未跨機）把多路請求併進同一輪；但 v2.0 每輪仍只有一輪在鏈上，要讓各段同時運算需要尚未撰寫的 v2.1。

## 實測結果

所有數字都來自兩台 NVIDIA N1X 筆電：**128 GB 節點**與 **64 GB 節點**，兩台都跑 WSL2（Ubuntu，NAT 模式），
用一條接在 USB4 埠之間、協商為 20 Gbps 的 Thunderbolt 3 線連接。估算值會特別標明。

### 連線（單一 TCP 連線；`linktest.py` 和 PowerShell）

| 路徑 | 吞吐量 | 來回延遲 p50（64 B） |
|---|---|---|
| Windows → Windows，走 USB4 | 7.1 Gbit/s | – |
| WSL（節點 A）→ Windows（節點 B） | 4.8 Gbit/s | 0.15 ms |
| 經 Windows 的 portproxy（iphlpsvc）進 WSL | 0.42 Gbit/s | 0.38 ms（p99 20 ms） |
| 同一台：Windows → WSL 的位址 | 16.1 Gbit/s | – |
| 同一台：經我們的使用者層中繼 | 10.1 Gbit/s | – |
| **WSL → USB4 → 中繼 → WSL（引擎實際走的路徑）** | **5.2 Gbit/s（615 MiB/s）** | **0.49 ms（p99 1.4 ms）** |

### Qwen3.8-Flash-Next：tfpp on TensorFold 0.6.1（已在硬體上驗證）

Checkpoint 為 `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`（48 層）。128 GB 節點跑嵌入層和第 0–31 層，64 GB 節點跑
第 32–47 層和輸出頭。Greedy 解碼，200 個 token，三個提示（繁中短文、附測試的 Python 函式、三點說明）。

| | 單機 TensorFold | 雙機 tfpp，無 KeepWarm | **雙機 tfpp，KeepWarm** |
|---|---|---|---|
| 輸出 token | 參考基準 | 3／3 逐字相同（serial、serial＋graph、MTP 6＋graph），MTP 接受數也相同 | **3／3 逐字相同** |
| Serial 解碼 | 32.3–32.8 tok/s（eager） | 24.5–25.1（eager）、25.1–25.5（graph）tok/s | **33.0–33.8 tok/s** |
| MTP 深度 6＋CUDA graph：短文／程式／說明文 | 31.4／77.1／38.2 tok/s | 23.3／60.3／29.4 tok/s | **31.0／79.3／38.6 tok/s** |
| 每輪時間 | – | 19.0 ms（128 GB 節點）＋15.5 ms（64 GB 節點）＋約 2 ms 連線與 Python＋1.7 ms leader commit | 17.4 ms（128 GB 節點）＋8.8 ms（64 GB 節點）＋約 2.5 ms 連線 |

- **為什麼需要 KeepWarm。** 同樣是第 32–47 層加輸出頭，在任一台上連續跑，每步都是 8.7–9.0 ms；先閒置 20–40 ms
  再跑，就變成 15–16.5 ms，因為閒置的 GPU 降頻了。等待時讓 CPU 空轉沒有用（15.9 ms）；在側邊 stream 放一個空轉
  的 GPU block 就恢復了（9.9 ms）。管線裡每一段大約有一半時間在等別段，所以每一段都被拖慢。llama.cpp RPC
  很可能也有同樣的損失。
- **兩台機器上跑三個 rank。** 128 GB 節點跑兩個 rank（第 0–15 層、第 16–31 層，本機連線），64 GB 節點跑第 32–47
  層。輸出 3／3 逐字相同、接受數相同，證明多 rank 的程式路徑沒問題。速度是 24.1／62.9／30.2 tok/s，偏低是因為
  兩個行程共用一張 GPU，不能代表三台實機的表現。

### Qwen3.8-Flash-Next：llama.cpp RPC（已在硬體上驗證）

unsloth 的 GGUF `UD-Q4_K_XL`（103.7 GiB），temperature 0。

| | 雙機 RPC（128 GB 節點約 80 GB，64 GB 節點約 20 GB） | 單機（128 GB），PLE 查表留在 SSD |
|---|---|---|
| 單路解碼（中文文章 530 token、程式 700 token） | **22.2 tok/s** | 22.8–23.6 tok/s |
| 9,014 token 的提示 | 760 tok/s（約 12 秒） | 691 tok/s |
| 載入時間 | 1 分 30 秒 | 24 秒 |
| `llama-server` 常駐記憶體 | 32 GB | 4.7 GB |
| 抽測（數學、程式、中文、推理） | 4／4 正確 | 4／4 正確 |

Flash-Next 的一個教訓：它的逐層 token 嵌入查表（PLE）在 Q4 檔裡佔 26.8 GiB，但每個 token 只讀 16 列、每列
90 bytes。用 `--lazy-mode on`，這張表就留在 SSD 上，Q4 模型一台 128 GB 節點就放得下，Flash-Next Q4 根本
不需要串聯。Q6_K（156.6 GiB，其中 105.9 GiB 要放 GPU 記憶體）才需要兩台。在單機上，TensorFold（serial
32.3–32.8 tok/s，開 MTP 草稿時程式碼最高 77 tok/s，見上表）比 llama.cpp（22.8–23.6 tok/s）快，主要差在 MTP
草稿和 kernel 效率；我們測試時，llama.cpp 還載入不了 Flash-Next 的 MTP 和 DFlash2 GGUF。

### GLM-5.3-Flash：tfpp-glm v1（已在截短模型上驗證）

底層：TensorFold v0.6.0 加上 MiaAI-Lab GLM 配方的 107 個 patch。Checkpoint 為
`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`。正確性測試用的是**只保留前 8 層的截短模型**，context 262,144，
開 DFlash2 草稿，19 個 bench 提示；每次都把每一輪的 token、保留數、草稿和 logits 雜湊拿去和單一行程比對。

| 測試 | 檢查什麼 | 結果 |
|---|---|---|
| T0a（pure） | 切開的計算對上套件原本的 `_compute` | 逐位元相同 |
| T0b | CUDA graph 對上 eager | 逐位元相同 |
| T1 | 兩段，切點 [0,4)+[4,8) 與 [0,5)+[5,8)，對上單一行程 | 兩種切點都逐位元相同 |
| T1-kept | kept prompt 在 0.5 GiB 記憶體壓力下 | 兩種切點的回覆都相同 |
| T1-fault | 注入例外：該請求中止、下一個請求仍相同；砍掉 follower：leader 以 75 結束 | 都符合規格 |
| 跨機 | 128 GB 節點 [0,4) → USB4 → 64 GB 節點 [4,8)＋輸出頭＋DFlash2，對上 64 GB 節點上的單一行程 | **19／19 逐位元相同** |
| 26 層跨機 | 切在 19 對上切在 20（這個大小沒有一張 GPU 放得下單一行程的參考） | **19／19 逐位元相同** |
| HTTP | leader 的 OpenAI 風格 API | 有回應 |

8 層模型的速度（開著 trace；只有 8 層的 GLM 輸出本來就是亂碼，只看速度和是否一致）：

| | 整體 | 每個解碼 token |
|---|---|---|
| 單一行程（64 GB 節點） | 65.1 tok/s | 11.7 ms |
| **跨機兩段** | **53.6 tok/s** | **14.0 ms** |
| 同一張 GPU 跑兩段（兩個行程搶 GPU，不具代表性） | 30.9 tok/s | 25.5 ms |

跨機每輪時間（關掉 trace）。26 層那次把第 [0,19) 或 [0,20) 層放在 128 GB 節點（57.3／60.8 GiB），其餘層加輸出頭
和 DFlash2 放在 64 GB 節點（26.5 GiB）：

| 每輪列數 | 8 層 | 26 層 | 每多一層 | 固定開銷 |
|---|---|---|---|---|
| R=1（一個 token） | 14.0 ms | 32.3 ms（31 tok/s） | 1.02 ms | 5.9 ms（輸出頭、取樣、一次跳轉、Python） |
| R=2（一個草稿） | 21.0 ms | 44.6 ms | 1.31 ms | 10.5 ms（含 DFlash2 草稿 4.2 ms） |

- 每層 1.02 ms 已經碰到記憶體頻寬上限：解碼每層要讀約 0.3 GB（8 個專家、共享專家、4-bit 注意力），N1X 約
  263 GB/s。
- 跨機每輪多約 2.3 ms，和 Flash-Next 的約 2.5 ms 一致。

**完整 45 層模型在三台上的推估（估算，未實測）：** 不開草稿時每個 token 約 54 ms（約 18.6 tok/s）。若 DFlash2
的接受率和雙 DGX Spark 上看到的一樣（58%，每輪約 2.6 個 token）：文章約 25–30 tok/s、程式約 35–40 tok/s。
對照組：**一對 DGX Spark（GB10）用 RoCE 跑張量平行**、搭配 MIA 的配方，實測單路 45.9／68.5 tok/s（文章／程式），
4 路 82.9／113.5 tok/s。管線上的單一請求同一時間只讓一張 GPU 工作，所以本質上大約是那一對機器的 55–65%；
要接近它的總吞吐量，需要多路請求加上重疊的輪次（tfpp-glm v2.1，尚未撰寫；v2.0 的模型推估約為單路的 1.5–1.7 倍）。

**DeepSeek-V4.1-Flash 推估（估算，什麼都還沒跑）：** 三台時每輪約 45–50 ms，約等於文章 25–37 tok/s、程式
40–55 tok/s。同一對 DGX Spark 上，MiaAI-Lab 配方公布的數字為文章 61.4、程式 89.3 tok/s（Zig 分支 deepseek-v41-zig；非我們實測）。

## 引擎與驗證狀態

| 引擎 | 模型 | 建立在 | 狀態 |
|---|---|---|---|
| llama.cpp RPC | Qwen3.8-Flash-Next，unsloth GGUF `UD-Q4_K_XL` | 上游 llama.cpp（`9febe63`，以 `GGML_RPC=ON` 編譯） | **已在硬體上驗證**：llama.cpp RPC 跑 Flash-Next Q4，雙機 22.2 tok/s |
| `tfpp` | Qwen3.8-Flash-Next，MLX 4-bit＋MTP | TensorFold 0.6.1 | **已在硬體上驗證**：雙機切分的輸出與單機逐 token 相同、速度一樣快（KeepWarm）；兩台機器上跑過三個 rank 的鏈。¹ |
| `tfpp-glm` v1（`PARALLEL=1`） | GLM-5.3-Flash，EXL3 4bpw（DFlash2 可選） | TensorFold v0.6.0＋MiaAI-Lab 的 107 個 patch | **已在硬體上驗證（截短模型）**：8 層模型與單一行程逐位元相同（pure、eager 與 graph、2 種切點、kept prompt 壓力、故障中止與砍行程、跨機 19／19）；26 層跨機兩種切點逐位元相同；HTTP 路徑有回應。² |
| `tfpp-glm` 完整模型 | GLM-5.3-Flash 全部 45 層 | 同上 | **尚未執行。** 完整 45 層模型的各段合計約需 160 GiB，128＋64 GB **放不下**，需要第三台（規劃中的鏈：128 GB → 64 GB → 128 GB）。 |
| `tfpp-glm` v2（`PARALLEL` 2–8） | 同上，多路 | 同上 | **尚未在硬體上執行。** 已寫好並做過靜態審查，只有 CPU 測試。 |
| `tfpp-ds41` | DeepSeek-V4.1-Flash-1017，EXL3 EAQ 2.9 bpw | BertholomusAI 的 TensorFold fork（`deepseek-v41-tp2`） | **尚未在硬體上執行。** 只做過靜態審查，加上它自己的 CPU 測試（都通過）；沒有在任何 GPU 上跑過。 |
| vLLM PP=2＋TensorFold kernel | MiniMax-M3.1-Flash，完整 60 層，EXL3 SAGE 2.15 bpw（108.6 GiB） | vLLM `minimax-m31-flash` image＋vllm-exl3（pack 作者的，AGPL-3.0）＋TensorFold v0.6.6 kernel | **已在硬體上驗證（2026-10-26）**：兩台機器、完整模型：不開草稿 17.8 tok/s；開 DFlash2 草稿器時，131,072 token 視窗下單路 14–44 tok/s（結構化文字最快）、1–3 路合計 31–36 tok/s；24/24 道可驗證題；96K token 的 needle 測試通過。呼叫 kernel 的膠水外掛因 AGPL 不公開，所以單靠本 repo 無法重現這些數字。詳見 [配方](docs/recipes/MiniMax-M3.1-Flash-SAGE-EXL3-2x-N1X-vLLM.md)。 |

¹ 硬體實測用的是第一版、沒有版本號的傳輸協定。程式之後改成 `tfpp-flashnext/2` 協定（連線握手、啟動時檢查
整條鏈、檢查每個 frame、錯誤 frame），這一版通過 CPU 測試，但還沒有重新上機。
² 硬體實測之後做的修正（列在 [`tfpp/tfpp/glm/IMPLEMENTATION_NOTES.md`](tfpp/tfpp/glm/IMPLEMENTATION_NOTES.md)）
只通過 CPU 測試。

這裡的「逐位元相同」指的是和**同一個引擎在 world size 1 下的單一行程**相同，不是和張量平行（TP=2）的部署相同：
TP 加總部分和的順序不同，位元本來就會不一樣。

各引擎的完整說明（所有參數，以及每個狀態背後的測試）：[llama.cpp RPC](docs/engines/llamacpp-rpc.md)、
[tfpp（Flash-Next）](docs/engines/tfpp-flashnext.md)、[tfpp-glm](docs/engines/tfpp-glm.md)、
[tfpp-ds41](docs/engines/tfpp-ds41.md)。所有實測數字與量測條件：[docs/benchmarks.md](docs/benchmarks.md)（英文）。

## 需要準備什麼

- **機器**：2–4 台有 NVIDIA GPU 的電腦，記憶體加起來要放得下模型。只在 NVIDIA N1X 筆電（GPU
  `sm_121`；128 GB 和 64 GB 統一記憶體）上測過。單台實測：記憶體讀取頻寬 263–285 GB/s，bf16 93 TFLOPS。其他
  GPU 和原生 Linux 節點都沒測過。
- **作業系統**：測試環境是 Windows 11＋WSL2（Ubuntu，aarch64，預設 NAT 網路）。所有運算都在 WSL 裡跑；Windows
  端只負責設定連線和跑中繼。
- **線材**：一條 Thunderbolt 4／USB4 線，**直接**插在兩台的 USB-C 埠上，不要經過擴充座。三台串成一條鏈時，
  中間那台需要兩個 USB4 埠、兩條線，每條線一個子網段。
- **實際可用的記憶體**：128 GB 節點大約可給權重和快取 92 GiB；64 GB 節點在 GPU 上不跑別的東西時約 30–40 GiB。
  Windows 至少留 4–6 GB。WSL 裡 CUDA 回報的 GPU 記憶體會跟著 WSL 的記憶體上限（`.wslconfig`）走：上限 108 GB 時
  為 104,475 MiB，上限 48 GB 時為 46 GiB（觀察到的現象，未證實）。
- **磁碟**：每段只需要自己那幾層的 shard，放在 WSL 的磁碟裡（不要放 `/mnt/c`）。別忘了 WSL 的虛擬磁碟是長在
  `C:` 上的。
- **軟體**（WSL 內）：Python 3.14（唯一上過機的版本）、CUDA 13 版的 PyTorch、pip 的 CUDA 13.0 工具鏈。每個
  引擎各用自己的 venv，見各自的 `toolchain.sh` 或 `env.sh`。

規劃切點用的每層大小：

| 模型 | 每層 | 固定部分 | 合計 |
|---|---|---|---|
| GLM-5.3-Flash EXL3 4bpw（45 層） | 3.67–3.70 GiB（MoE 層）、0.54 GiB（第 0–2 層 dense） | 嵌入 1.18、輸出頭 1.18、DFlash2 常駐約 0.9（每段另有快取和工作緩衝） | 磁碟上 163.6 GiB；各段合計約 160 GiB |
| DeepSeek-V4.1-Flash EXL3 EAQ 2.9 bpw（40 層） | 4.86 GiB（第 18–22 層為 3.28 GiB） | 嵌入 1.23、輸出頭 0.46、DSpark 草稿器 6.75 | 196.1 GiB，另有兩張各 94.6 GiB 的 Engram 表，從 SSD 讀取 |

## 快速開始

在每台節點的 WSL 裡，把這個 repo clone 到 `~/hive-ram`。tfpp-glm 的 systemd unit 假設是這個路徑；tfpp-ds41 的
情況見下方說明。除非另外註明，WSL 裡的指令都在 `~/hive-ram` 下執行。

先跑 CPU 測試（不需要 GPU，也不需要 TensorFold；需要 pytest 和 numpy）：

```bash
python3 -m venv ~/.venvs/hive-ram-test && . ~/.venvs/hive-ram-test/bin/activate && pip install -e ~/hive-ram"[test]"
cd ~/hive-ram/tfpp
python -m pytest -q tests scripts/tests tfpp/glm/tests tfpp/ds41/tests
```

### 步驟 0：連接兩台機器（所有引擎都要）

完整步驟（含三台串聯與 WSL 設定）見 [docs/hardware-setup.md](docs/hardware-setup.md)；遇到問題時見
[docs/troubleshooting.md](docs/troubleshooting.md)（皆為英文）。

線先插好，在**每一台** Windows 上開**系統管理員**身分的 PowerShell：

```powershell
# 節點 1 設為 10.77.0.1，只對對方開放 50052（llama.cpp RPC）和 5201（連線測試、tfpp）
powershell -ExecutionPolicy Bypass -File setup_usb4_link.ps1 -Node 1
# 節點 2 設為 10.77.0.2
powershell -ExecutionPolicy Bypass -File setup_usb4_link.ps1 -Node 2
```

它只會在 USB4 網卡上設定固定 IP（網路類別設為「私人」），並加上防火牆規則，**只允許對方的位址**連進 ICMP 和
指定的 TCP port。其他連線用 `-Address`、`-PeerAddress`、`-AllowTcpPorts`、`-InterfaceAlias` 指定（腳本開頭有
三台的範例）。要還原就執行 `undo_usb4_link.ps1`。在步驟 0、A、B 中，節點 1 是接受連線、跑後段層的那台（我們
實測時是 64 GB 節點），節點 2 跑前段層並帶動整條鏈（128 GB 節點）。tfpp-glm 和 tfpp-ds41 則從 `node.env`
讀取位址。

在**接收端**節點的 Windows 上（不需要系統管理員）啟動往 WSL 的中繼，視窗要保持開著：

```powershell
powershell -ExecutionPolicy Bypass -File relay_user.ps1      # 監聽 10.77.0.1 的 50052 和 5201
# 其他位址／port：-ListenAddress 10.77.0.2 -Ports 5301      手動指定 WSL 位址：-WslIp <位址>
```

WSL 每次重啟後 NAT 位址會變，中繼也要重新啟動。如果舊的設定在這些 port 上留下 `netsh portproxy` 項目，請用系統
管理員身分執行一次 `start_n1x_relay.ps1`：它會先刪掉那些項目，再啟動中繼。

測試真正的路徑（WSL 到 WSL）：

```bash
python3 linktest.py server           # 節點 1 的 WSL 內（服務一個 client 後結束）
python3 linktest.py client 10.77.0.1 # 節點 2 的 WSL 內
```

正常應該約 5 Gbit/s、來回延遲 p50 約 0.5 ms。如果 p99 高達數十 ms，表示流量走的是 portproxy 而不是中繼。

### A. llama.cpp RPC（不需要 hive-ram 的程式）

兩台都在 WSL 裡編譯含 CUDA 和 RPC 的 llama.cpp。我們用 CUDA 13.1 編譯 `9febe63` 這個 commit：

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCUDAToolkit_ROOT=/usr/local/cuda-13.1   # 不加的話，我們第一次編譯連到了系統裡較舊的 CUDA
cmake --build build -j
```

節點 1（在 `relay_user.ps1` 後面，中繼轉送 50052）跑後段層的 RPC 伺服器。新版叫 `ggml-rpc-server`，舊版叫
`rpc-server`，參數請以你那一版的 `--help` 為準。不要開它的本機快取（`-c`）：寫快取檔也會吃掉 WSL 的記憶體。

```bash
build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052
```

節點 2 放 GGUF 檔，跑伺服器和前段層。這是實測時用的指令：

```bash
build/bin/llama-server -m <dir>/<model>-UD-Q4_K_XL-00001-of-00004.gguf --rpc 10.77.0.1:50052 \
  -ngl 999 --fit on --fit-target 6144,4096 -c 32768 -np 1 -fa on -lm dio -cram 2048 \
  --host 127.0.0.1 --port 18080
```

- `-lm dio` 用 direct I/O 讀權重，頁快取就不會吃掉模型需要的記憶體。
- Flash-Next 建議先試單機：`-lm dio --lazy-mode on -ngl all --fit off`。ggml-cuda 把 N1X 當成內顯，所以
  `--lazy-mode auto` 會自己關掉；而 `-lm mmap` 不開 lazy 時，WSL 每讀一列 90 bytes 就會預讀 16 MB。把伺服器放在
  cgroup 裡跑（例如 `systemd-run ... -p MemoryHigh=8G`）也能壓住頁快取。

### B. tfpp：Qwen3.8-Flash-Next on TensorFold 0.6.1

每台節點建一次 venv：步驟寫在 `tfpp/scripts/flashnext/env.sh` 開頭（TensorFold `v0.6.1`、CUDA 13 版的 torch
2.14.1、pip 的 CUDA 13.0）。注意 `env.sh` 是依實測環境重建的，本身還沒有在硬體上跑過。每台需要 `config.json`、
tokenizer 和自己那幾層的 shard；最後一個 rank 還需要輸出頭和 MTP 的 shard。最簡單的做法是兩台都放整份
checkpoint。

```bash
cd ~/hive-ram/tfpp && . scripts/flashnext/env.sh
# 節點 1（10.77.0.1，在 relay_user.ps1 後面）：最後一個 rank
python tfpp_run.py --role last --model <dir> --layers 32:48 --listen 5201 --depth 6 --graphs
# 節點 2：leader（rank 0）；等整條鏈接好後跑三個提示，並寫出 token
python tfpp_run.py --role leader --model <dir> --layers 0:32 --peer 10.77.0.1:5201 --depth 6 --graphs --out pp.json
# 單機參考（整個模型要放得進一台），再逐 token 比對
python tfpp_run.py --role local --model <dir> --depth 6 --graphs --out ref.json
python compare.py ref.json pp.json
```

- 每個 rank 的 `--depth` 和 `--capacity` 要一致。中間的 rank 要給 `--layers`、`--listen` 和 `--peer`。各 rank
  啟動順序不拘。
- 可選設定：`TFPP_SECRET`（每台設同樣的值，每條連線的兩端都要證明自己知道它）、`TFPP_WARM_MS`（KeepWarm 每次
  空轉的長度，預設 1.0；設 0 關閉）、`TFPP_TIMING=1`（各階段計時）、`TFPP_IO_TIMEOUT`（預設 600 秒）。
- `rank_bench.py` 單獨量一個 rank 每步的時間，可以加上閒置間隔，用來觀察 GPU 降頻的影響。
- `tfpp_run.py` 是量測與驗證用的程式：跑一組固定的提示，沒有 HTTP 伺服器。

### C. tfpp-glm：GLM-5.3-Flash on MIA 套件

**1. venv 和底層套件**（每台節點，WSL 內；完整清單見 `tfpp/scripts/glm/toolchain.sh`）：

```bash
python3.14 -m venv ~/tf-venv-mia && . ~/tf-venv-mia/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0"
pip install "triton==3.7.1" ninja "cuda-toolkit[nvcc,cccl]==13.0.*" numpy safetensors \
    "tokenizers>=0.20" "jinja2>=3.1" "huggingface-hub>=0.24"
# 只有 leader 需要：pip install "torchvision==0.28.0" Pillow "av==18.1.0" "xgrammar==0.2.8" "transformers>=5.3,<6"
rsync -a --delete "$MIA_TREE"/tensorfold/ \
    "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/tensorfold/"
cd ~/hive-ram/tfpp && python -m tfpp.glm.pins --check      # 必須回報樹雜湊 739ce991...
```

`$MIA_TREE` 是 TensorFold `v0.6.0` 依檔名順序套上 MiaAI-Lab 公開配方 `GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`
（commit `b5808ac`，「v1.11-pre」）的 107 個 patch。把這些 patch 套到 v0.6.0 tag 上，
離線重算可得到釘選的雜湊（見 [`THIRD_PARTY.md`](THIRD_PARTY.md)）。只要樹不同，每一段都會拒絕啟動。

**2. 權重。** 使用 `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`（83 個 shard，163.6 GiB）。每段需要
`config.json`、tokenizer、index 和自己的 shard；`node.env.example` 列出了它那組切點各段要的 shard 編號。
DFlash2 草稿器 `incoai/GLM-5.3-Flash-DFlash2@bf582e4e` **是可選的，授權為 CC BY-NC-ND 4.0（非商業、禁止改作）**：
設 `NO_DRAFTS=1` 就不用它（逐 token 解碼）。

**3. 單 GPU 等價測試**，用截短成 8 層的模型（我們的實測在 64 GB 節點上跑；兩台都放得下）：

```bash
cd ~/hive-ram/tfpp
export MODEL_DIR=$HOME/models/glm53 DRAFTER_DIR=$HOME/models/glm53-dflash2   # DRAFTER_DIR 留空即 --no-drafts
L=8 CONTEXT=2048 scripts/glm/run_truncated.sh all       # T0a、T0b、T1（兩種切點）、T1-kept、T1-fault
L=8 CONTEXT=2048 scripts/glm/run_truncated.sh T1-http
```

通過標準：`python -m tfpp.glm.compare ref.jsonl got.jsonl` 全部回報 IDENTICAL。

**4. 跨機。** 每台把 `tfpp/scripts/glm/node.env.example` 複製成 `node.env`，只保留這台的那一段（角色、層範圍、
模型資料夾、`DOWN`／`LISTEN` 位址）。`cluster.env` 放的是每一段都必須一致的設定；啟動時各段會比對這些設定的
雜湊，不一致就拒絕執行。在每一台接收連線的節點的 Windows 端，先（以系統管理員 PowerShell）放行對方連到這個 stage port，再啟動中繼。兩台、在 10.77.0.1 接收（和我們的實測一樣；
三台請用 docs/hardware-setup.md 裡 -Address／-PeerAddress／-AllowTcpPorts 的寫法）：

```powershell
powershell -ExecutionPolicy Bypass -File setup_usb4_link.ps1 -Node 1 -AllowTcpPorts 50052,5201,5301
powershell -ExecutionPolicy Bypass -File tfpp\scripts\glm\relay_hop.ps1 -ListenAddress 10.77.0.1 -Ports 5301
```

接著在每台的 WSL 內執行（順序不拘）：

```bash
cd ~/hive-ram/tfpp/scripts/glm
./start_stage.sh --dry-run    # 只做檢查：venv、套件雜湊、shard、記憶體、port；印出要執行的指令
./start_stage.sh              # 背景執行；leader 會等到開始服務並通過冒煙測試
```

leader 提供 OpenAI 風格的 API（`cluster.env` 的 `HTTP_HOST`／`HTTP_PORT`，預設 port 8888）：

```bash
curl -s http://127.0.0.1:8888/v1/models
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

- 要停止時，在 **leader** 上執行 `./stop_stage.sh`；QUIT 會沿著鏈傳下去。任何一段斷線都會以 75 結束，之後整條鏈
  都要重新啟動。
- `HTTP_HOST=0.0.0.0` 會在所有網路介面上提供**沒有驗證**的 API。除非那個 port 有防火牆擋著，否則請用
  `127.0.0.1`。
- 同一台上還有其他 GPU 伺服器時：把它們的 port 寫進 `GPU_GUARD_PORTS`，`start_stage.sh` 就不會在它們旁邊啟動。
  `VERIFY_SHARDS=1` 會先用 Hub 的 sha256 檢查這一段的 shard。
- 記憶體不夠放完整模型時，截短的跨機測試是在 leader 的 `node.env` 用 `TRUNCATE`、`TAP_LAYERS`、`BENCH`、
  `TRACE` 這幾個設定來跑（見 `node.env.example`）。
- 多路的 v2（在 leader 的 `node.env` 設 `PARALLEL=4` 等，測試在 `scripts/glm/run_v2.sh`）還沒有在硬體上跑過。

### D. tfpp-ds41：DeepSeek-V4.1-Flash on BertholomusAI fork（尚未上機）

這部分完全沒有在 GPU 上跑過。請把它當成第一次上機的除錯過程，而不是現成的配方。腳本預期的順序如下（細節見
[`tfpp/tfpp/ds41/IMPLEMENTATION_NOTES.md`](tfpp/tfpp/ds41/IMPLEMENTATION_NOTES.md)）：

```bash
# 1. venv ~/tf-venv-dsv41 和 fork 的 LF 原始碼：步驟在 tfpp/scripts/ds41/toolchain.sh 開頭
#    （對 github.com/bertholomus/TensorFold 的 deepseek-v41-tp2 分支、commit af26d42b1 做 git archive）
cd ~/hive-ram/tfpp
python -m tfpp.ds41.pins --check <fork-src>/tensorfold   # 必須回報 c64d3e0a...
python -m tfpp.ds41.prebuild                              # 編譯四個擴充，並檢查它們連結的函式庫
# 2. 11 層的測試模型 S11（59.9 GiB）：印出下載腳本、自己執行，再在每台建立 S11
scripts/ds41/download_list.sh > get_s11.sh   # 會先檢查磁碟空間；改好腳本裡的 REV 那一行再執行
SRC=$HOME/models/ds41-src scripts/ds41/make_testmodel.sh
# 3. 單 GPU，依這個順序；每一步都要 IDENTICAL 才能做下一步
export MODEL_DIR=$HOME/models/ds41-s11
scripts/ds41/run_shortened.sh T-ref
scripts/ds41/run_shortened.sh T0a        # 接著 T0b、T0c、T1-oracle、T1-exact、T1-long、T1-multi、T1-fault、T1-http
# 4. 雙機：node.env 的測試區塊（T1-c6、T1-c3、T1-3）；接收端先（系統管理員）`setup_usb4_link.ps1 -Node 1
#    -AllowTcpPorts 5321`，再 relay_hop.ps1 -Ports 5321；每台執行 start_stage.sh
```


## Repo 結構

```
hive-ram/
├── README.md、README.zh-TW.md
├── LICENSE、NOTICE、THIRD_PARTY_NOTICES.md
├── THIRD_PARTY.md              如何依釘選版本取得第三方程式樹並檢查雜湊
├── SECURITY.md                 威脅模型：連線與 API 保護了什麼、沒保護什麼
├── docs/                       說明與設計文件（索引：docs/README.md）
├── setup_usb4_link.ps1         Windows，需系統管理員：為一條 USB4 連線設固定 IP＋只允許對方的防火牆規則
├── undo_usb4_link.ps1          Windows，需系統管理員：移除 setup_usb4_link.ps1 加上的設定
├── relay_user.ps1              Windows，不需系統管理員：從連線位址轉進 WSL 的使用者層 TCP 中繼
├── start_n1x_relay.ps1         Windows，需系統管理員：先刪掉舊的 portproxy 項目，再執行 relay_user.ps1
├── linktest.py                 量測真實路徑的吞吐量與來回延遲（只用標準函式庫）
└── tfpp/
    ├── tfpp_run.py             Flash-Next：單 GPU，或管線中的一個 rank（量測用）
    ├── compare.py              Flash-Next：逐 token 比對兩次執行結果
    ├── rank_bench.py           Flash-Next：單一 rank 每步的時間，可加閒置間隔（觀察 GPU 降頻）
    ├── tfpp/                   Flash-Next 管線：load、part、driver、wire、proto、warm（KeepWarm）
    │   ├── glm/                tfpp-glm v1 與 v2（v2*.py）、IMPLEMENTATION_NOTES.md、schedules/、tests/
    │   └── ds41/               tfpp-ds41、IMPLEMENTATION_NOTES.md、tests/
    ├── scripts/
    │   ├── flashnext/env.sh    Flash-Next 的環境設定
    │   ├── glm/                toolchain、cluster.env、node.env.example、start/stop_stage.sh、relay_hop.ps1、
    │   │                       run_truncated.sh、run_v2.sh、systemd unit
    │   ├── ds41/               tfpp-ds41 的同一組腳本，另有 download_list.sh、make_testmodel.sh、run_shortened.sh
    │   └── tests/              連線工具與腳本的 CPU 測試
    └── tests/                  Flash-Next 管線的 CPU 測試
```

不在這個 repo 裡的東西：TensorFold、MIA 的 patch 組、BertholomusAI 的 fork（請自行依釘選版本取得，見
[`THIRD_PARTY.md`](THIRD_PARTY.md)；tfpp-glm 和 tfpp-ds41 啟動時會檢查雜湊），以及任何模型權重。

## 限制與注意事項

**目前做不到的事**
- **增加的是容量，不是速度。** 單路的速度大約等於一台機器讀完全部權重的速度。tfpp-glm v1 和 Flash-Next 測試程式
  一次只服務一個請求。
- **完整的 GLM-5.3-Flash 從沒跑過。** 128＋64 GB 放不下。DeepSeek-V4.1-Flash 則完全沒跑過：它的權重（196.1 GiB）
  就算三台也很緊，而同時擁有兩個 Engram 層的那一段需要約 273 GiB 磁碟（自己的權重加兩張 Engram 表），超過我們
  節點的剩餘空間。
- **只在一種機器上測過**：N1X 筆電（`sm_121`）、WSL2 NAT、Python 3.14。程式接受任意段數，但實際最多只用過兩台
  機器（三個 rank）。
- **上游版本是釘死的。** 每個 tfpp 版本都會在執行時替換某一棵特定 TensorFold 樹裡的指定函式。tfpp-glm 和
  tfpp-ds41 遇到別的樹就拒絕啟動（雜湊檢查）；Flash-Next 的 tfpp 只檢查它要替換的那一行載入程式還在，所以請確實
  用 TensorFold `v0.6.1`。上游出新版需要重新審查，不是 pip 升級就好。
- **一壞全停。** 任何一段出錯，整條鏈都會停（結束碼 75），所有段都要重新啟動。

**安全性**
- 各段之間的連線沒有加密。tfpp-glm、tfpp-ds41 和 llama.cpp RPC 都沒有驗證機制；Flash-Next 的連線可以要求共用的
  `TFPP_SECRET`。llama.cpp 的 RPC 伺服器允許任何 client 讀寫它的緩衝區。
- 所有 port 都只開在點對點的線上：綁定連線位址、只允許對方連入（`setup_usb4_link.ps1` 就是這樣設的）。中繼轉送
  到的 WSL 內部位址，同一台 Windows 上的任何程式都連得到。
- tfpp-glm 的 API 沒有驗證。tfpp-ds41 的 leader 可以要求金鑰（在它的 `node.env` 設 `TF_API_KEY_FILE`）。
- 威脅模型與強化步驟見 [`SECURITY.md`](SECURITY.md)。

**我們遇過的 Windows 與 WSL 問題**
- USB4 網卡在一台筆電上出現了卻一直顯示「未連線」，另一台則根本沒出現。那台筆電重開機後就好了；韌體和更新版本的
  差異最後證實不是原因。線要直接插在兩台之間，不要經過擴充座。
- 驅動程式和作業系統更新可能自動安裝，並在實驗途中重開機。實驗期間請暫停自動更新。有一次韌體更新還改變了 64 GB
  筆電在 Windows 和 GPU 之間分配記憶體的方式（更新後 Windows 只看得到 32 GB），所以每次更新後都要重新確認可用
  記憶體。
- WSL 的頁快取會算進 Windows 的記憶體認可量（commit）。一次大量下載後，某台節點留下 77 GB 的頁快取，Windows
  可用的 commit 只剩 1.2 GB，分頁檔一路長到 `C:` 滿為止。大量複製檔案後請清掉快取
  （`echo 1 | sudo tee /proc/sys/vm/drop_caches`）。tfpp-glm 的載入器用 `O_DIRECT`，並在載入前要求核心把每個
  shard 從快取中丟掉。
- 同一台上的其他 GPU 程式會佔用同一塊統一記憶體。載入一段之前請先停掉它們。
- 筆電功耗 45–80 W，長時間滿載會很熱：請接著電源，並留意溫度。

## 路線圖

1. **第三台節點**（128 GB），用三台跑完整的 45 層 GLM-5.3-Flash。規劃的鏈是 128 GB → 64 GB → 128 GB；範例檔
   `tfpp/scripts/glm/node.env.example` 附的是另一種排法 64 GB → 128 GB → 128 GB（切點 [0,7)／[7,26)／[26,45)）。
   實際順序會在上機時依各台真正的可用記憶體決定。
2. **tfpp-glm v2 上機**：多路請求同時走在鏈上（`PARALLEL` 2–8），先單 GPU，再兩台、三台。之後是 v2.1：讓各段
   重疊運算（分組管線），提高總吞吐量。
3. **tfpp-ds41 第一次上機**：先在單 GPU 上跑 11 層測試模型，再跑兩台；之後在記憶體和 SSD 空間足夠放 Engram 表時
   跑完整模型。
4. 把實測之後做的修正重新上機驗證，並加上一支依釘選版本下載、驗證第三方程式樹的腳本。
5. 實驗：用 WSL mirrored 網路在 USB4 上跑 NCCL 張量平行（為了量測，不是因為預期它會贏），以及在 Python 版驗證
   完成後移植到 TensorFold 的原生（Zig）引擎。

細節與上機步驟的先後順序見 [docs/roadmap.md](docs/roadmap.md)（英文）。

## 授權

- hive-ram 自己的程式碼與文件：**Apache License 2.0**，著作權屬於 hive-ram contributors。見 [`LICENSE`](LICENSE)
  和 [`NOTICE`](NOTICE)。tfpp 有部分改寫自 TensorFold 原始碼，tfpp-glm 和 tfpp-ds41 也依照 MIA 套件和
  BertholomusAI fork 的程式，在執行時替換其中的函式；改寫了哪些，`NOTICE` 有列出。
- tfpp 建立在 **TensorFold**（Apache-2.0；0.6.0 之前的程式碼為 MIT）、**MiaAI-Lab 的 GLM 配方 patch**、
  **BertholomusAI 的 TensorFold fork**（Apache-2.0）和 **ExLlamaV3** 的 EXL3 格式（MIT）之上。llama.cpp／ggml
  （MIT）是透過 RPC 當作獨立程式使用，沒有複製進來。這些程式樹都不隨本 repo 發布。致謝與授權全文見
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。本 repo 沒有新增、import 或複製任何 AGPL 程式碼（MiniMax 配方會執行該 pack 作者 image 裡的 AGPL-3.0 vllm-exl3 外掛，那是你自己安裝的獨立程式，hive-ram 不含任何 import 它的程式）。
- **本 repo 不包含任何模型權重。** 每個 checkpoint 各有自己的授權：下載前、以及任何非個人用途之前，請先看模型卡。
  特別是 GLM 可選的 DFlash2 草稿器（`incoai/GLM-5.3-Flash-DFlash2`）採 **CC BY-NC-ND 4.0：限非商業使用、禁止
  改作**。請勿再散布它或它的轉換版本。不用它的方法：設 `NO_DRAFTS=1`；或在 `PARALLEL=1` 時用 checkpoint 自帶的
  MTP 頭（`serve.py` 不給 `--drafter`；`start_stage.sh` 沒有接這個選項，也沒有在硬體上跑過；MIA 的配方把這個
  模式叫 `DRAFTER=mtp`，只能單路，速度約慢 5–10%）。本專案不涵蓋 DFlash2 的商業用途：如果你的用途是商業性的，
  請不要用它。
- NVIDIA、N1X、DGX Spark、GB10、Thunderbolt、USB4、DeepSeek、Qwen 和 GLM 是其各自所有者的商標，
  這裡只用來描述硬體與模型。

## 致謝

感謝 TensorFold 及其貢獻者；MiaAI-Lab（Hugging Face 上為 Mia-AiLab）提供 GLM-5.3-Flash 配方和 EXL3 量化；
BertholomusAI 的 DeepSeek-V4.1 TensorFold fork；turboderp 的 ExLlamaV3；llama.cpp／ggml 專案；以及 unsloth 的
GGUF 量化。
