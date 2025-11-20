mod comms;
mod consensus;
mod crypto;
mod inference;
mod topology;
mod types;

use crate::comms::{CommsConfig, CommsHandle, OutEvent};
use crate::consensus::{ConsensusConfig, ConsensusEngine, SignedGossip};
use crate::crypto::{CryptoConfig, CryptoSuite};
use crate::inference::{InferenceConfig, InferenceEngine};
use crate::topology::{TopologyConfig, TopologySelector};
use crate::types::{GeoPoint, GgsMessage};
use anyhow::Result;
use futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use std::sync::Arc;
use tokio::time::{interval, Duration};

struct AppConfig {
    inference: InferenceConfig,
    comms: CommsConfig,
    topology: TopologyConfig,
    crypto: CryptoConfig,
    consensus: ConsensusConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig::default(),
            comms: CommsConfig::default(),
            topology: TopologyConfig::default(),
            crypto: CryptoConfig::default(),
            consensus: ConsensusConfig::default(),
        }
    }
}

struct Node {
    comms: CommsHandle,
    inference: InferenceEngine,
    topology: TopologySelector,
    consensus: ConsensusEngine,
    tick_counter: u64,
}

impl Node {
    async fn new(config: AppConfig) -> Result<Self> {
        let mut rng = rand::thread_rng();
        let geo = GeoPoint::random(&mut rng);
        let inference = InferenceEngine::new(config.inference)?;
        let comms = CommsHandle::new(config.comms).await?;
        let topology = TopologySelector::new(geo.clone(), config.topology);
        let crypto_suite = Arc::new(CryptoSuite::new(config.crypto)?);
        let consensus = ConsensusEngine::new(crypto_suite.clone(), config.consensus);
        println!(
            "启动 GGS 节点 => peer: {}, eth {}, sol {} @ ({:.2},{:.2})",
            comms.peer_id,
            crypto_suite.eth_address(),
            crypto_suite.sol_address(),
            geo.lat,
            geo.lon
        );
        println!("模型维度: {}", inference.model_dim());
        Ok(Self {
            comms,
            inference,
            topology,
            consensus,
            tick_counter: 0,
        })
    }

    async fn run(mut self) -> Result<()> {
        let mut ticker = interval(Duration::from_secs(10));
        loop {
            tokio::select! {
                event = self.comms.swarm.select_next_some() => {
                    if let SwarmEvent::Behaviour(out) = event {
                        self.handle_network_event(out).await?;
                    }
                }
                _ = ticker.tick() => {
                    self.on_tick().await?;
                }
            }
        }
    }

    async fn on_tick(&mut self) -> Result<()> {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        let hash = self.inference.tensor_hash();
        let heartbeat = GgsMessage::Heartbeat {
            peer: self.comms.peer_id.to_string(),
            model_hash: hash,
        };
        self.publish_signed(heartbeat).await?;

        let embedding = self.inference.embedding();
        let probe = GgsMessage::SimilarityProbe {
            embedding,
            position: self.topology.position(),
            sender: self.comms.peer_id.to_string(),
        };
        self.publish_signed(probe).await?;

        self.inference.local_train_step();
        self.consensus.prune_stale();
        if self.tick_counter % 12 == 0 {
            self.maybe_broadcast_dense().await?;
        }
        self.check_topology_health();
        Ok(())
    }

    async fn handle_network_event(&mut self, event: OutEvent) -> Result<()> {
        match event {
            OutEvent::Gossipsub(g) => {
                if let libp2p::gossipsub::Event::Message {
                    propagation_source,
                    message,
                    ..
                } = g
                {
                    if let Ok(signed) = serde_json::from_slice::<SignedGossip>(&message.data) {
                        if self.consensus.verify(&signed) {
                            self.handle_signed_message(signed, propagation_source.to_string())
                                .await?;
                        } else {
                            eprintln!("签名验证失败，来自 {:?}", propagation_source);
                        }
                    }
                }
            }
            OutEvent::Mdns(event) => {
                if let libp2p::mdns::Event::Discovered(peers) = event {
                    for (peer, _addr) in peers {
                        println!("通过 mDNS 发现节点 {peer}");
                    }
                }
            }
        }
        Ok(())
    }

    async fn publish_signed(&mut self, payload: GgsMessage) -> Result<()> {
        let signed = self.consensus.sign(payload)?;
        self.comms.publish(&signed)?;
        if !self.comms.broadcast_realtime(&signed).await {
            println!("[FAILOVER] QUIC 广播失败，已回落到纯 Gossip");
        }
        Ok(())
    }

    async fn handle_signed_message(&mut self, signed: SignedGossip, source: String) -> Result<()> {
        match &signed.payload {
            GgsMessage::Heartbeat { peer, .. } => {
                self.consensus.update_stake(peer, 0.0, 0.0, 0.05);
                println!("收到 {} 的心跳 (via {source})", peer);
            }
            GgsMessage::SimilarityProbe {
                embedding,
                position,
                sender,
            } => {
                let self_embedding = self.inference.embedding();
                self.topology.update_peer(
                    sender,
                    embedding.clone(),
                    position.clone(),
                    &self_embedding,
                );
                if let Some(snapshot) = self.topology.peer_snapshot(sender) {
                    let stake = self.consensus.stake_weight(sender);
                    println!(
                        "拓扑更新：{} => sim {:.3}, geo {:.3}, stake {:.3}, dim {}, pos ({:.1},{:.1})",
                        sender,
                        snapshot.similarity,
                        snapshot.geo_affinity,
                        stake,
                        snapshot.embedding_dim,
                        snapshot.position.lat,
                        snapshot.position.lon
                    );
                }
                if self.should_send_sparse_update(sender) {
                    if self.comms.allow_sparse_update() {
                        let update = self.inference.make_sparse_update(16);
                        let msg = GgsMessage::SparseUpdate {
                            update,
                            sender: self.comms.peer_id.to_string(),
                        };
                        self.publish_signed(msg).await?;
                    } else {
                        println!("[带宽限制] 本轮跳过稀疏更新");
                    }
                }
            }
            GgsMessage::SparseUpdate { sender, update } => {
                self.inference.apply_sparse_update(update);
                self.consensus.update_stake(sender, 0.1, 0.0, 0.1);
                println!("应用来自 {} 的稀疏更新", sender);
            }
            GgsMessage::DenseSnapshot { snapshot, sender } => {
                self.inference.apply_dense_snapshot(snapshot);
                self.consensus.update_stake(sender, 0.0, 0.2, 0.05);
                println!("融合 {} 的模型快照", sender);
            }
        }
        Ok(())
    }

    fn should_send_sparse_update(&self, target: &str) -> bool {
        let primary = self.topology.select_neighbors();
        if primary.iter().any(|peer| peer == target) {
            return true;
        }
        self.topology.mark_unreachable(target);
        false
    }

    fn check_topology_health(&self) {
        let (primary, backups) = self.topology.neighbor_sets();
        if primary.len() < self.topology.max_neighbors() && !backups.is_empty() {
            println!(
                "[拓扑 Failover] 主邻居 {}/{}，启用备份 {:?}",
                primary.len(),
                self.topology.max_neighbors(),
                backups
            );
        } else if backups.len() < self.topology.failover_pool() {
            println!(
                "[拓扑提示] 备份邻居不足 {}/{}",
                backups.len(),
                self.topology.failover_pool()
            );
        }
    }

    async fn maybe_broadcast_dense(&mut self) -> Result<()> {
        let snapshot = self.inference.tensor_snapshot();
        let bytes = snapshot.values.len() * std::mem::size_of::<f32>();
        if self.comms.allow_dense_snapshot(bytes) {
            let msg = GgsMessage::DenseSnapshot {
                snapshot,
                sender: self.comms.peer_id.to_string(),
            };
            self.publish_signed(msg).await?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::default();
    let node = Node::new(config).await?;
    node.run().await
}
