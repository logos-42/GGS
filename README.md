# GGS 去中心化训练节点

面向 Geo-Similarity-Weighted Gossip (GGS) 的 Rust 节点实现，集成真实推理张量、QUIC Gossip 通道、地理 + 嵌入双指标拓扑、以及 Web3 签名 / 质押 / 信誉系统，可直接部署到 Base 网络环境。

## 功能概览

- **推理引擎 (`src/inference.rs`)**
  - 从 `.npy` 模型参数加载，维护 TensorSnapshot、SparseUpdate，并带 residual 误差反馈。
  - 支持 Top-K 稀疏更新、密集快照、local training tick；可输出模型 hash & 维度。

- **通信层 (`src/comms.rs`)**
  - 基于 `libp2p gossipsub + mDNS` 的控制平面。
  - QUIC (`quinn`) 数据平面，带带宽预算（稀疏次数 / 密集字节）和 failover 回落。

- **拓扑模块 (`src/topology.rs`)**
  - Geo + embedding 双指标评分，维护主邻居 + 备份池，支持 failover / mark unreachable。
  - 为日志提供 `PeerSnapshot`（相似度、地理亲和、嵌入维度、位置）。

- **共识与 Web3 (`src/consensus.rs`, `src/crypto.rs`)**
  - 以太坊 (k256) + Solana (ed25519) 双签名；stake/reputation 计分。
  - 心跳 / 稀疏 / 密集消息统一签名与验证，并按活动自动调整信誉。

## 快速开始

```bash
cargo check          # 仅编译检查
cargo run            # 运行节点，默认随机Geo位置 & 128维模型
```

启动日志中将输出本地 peer id、ETH/SOL 地址、模型维度，以及拓扑评分详情。默认 Gossip 主题为 `ggs-training`，可在 `CommsConfig` 自定义监听地址 / QUIC 端口 / 带宽预算。

## 自定义与扩展

- 在 `Cargo.toml` 中加入所需的推理库（如 Candle、ONNX Runtime）后，扩展 `InferenceEngine` 的加载逻辑即可。
- 若要使用实际的链上 RPC，可在 `ConsensusEngine` 中替换当前内存 staking 账本。
- 通过 `TopologyConfig` 的参数可调邻居数量、备份池大小、地理缩放等策略。

## 目录结构

```
├── src/
│   ├── main.rs          # Node 入口，驱动所有模块
│   ├── comms.rs         # Gossip + QUIC 通信层
│   ├── inference.rs     # 推理张量与更新逻辑
│   ├── topology.rs      # 拓扑评分与 failover
│   ├── consensus.rs     # 签名、质押、信誉
│   └── crypto.rs        # ETH/SOL 密钥管理
├── Cargo.toml
└── README.md
```

## 贡献与远程

仓库：`git@github.com:logos-42/GGS.git`  
当前默认分支为 `master`，欢迎在此基础上提交 PR 或扩展模块（例如 WebRTC、治理合约集成等）。


