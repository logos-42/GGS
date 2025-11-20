use crate::crypto::{CryptoSuite, SignatureBundle};
use crate::types::GgsMessage;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct StakeRecord {
    pub stake_eth: f64,
    pub stake_sol: f64,
    pub reputation: f64,
    pub last_seen: Instant,
}

impl StakeRecord {
    pub fn combined_weight(&self) -> f32 {
        let stake_component = (self.stake_eth + self.stake_sol).ln_1p() as f32;
        let rep_component = (self.reputation.max(0.0) as f32).ln_1p();
        (stake_component + rep_component).clamp(0.0, 5.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignedGossip {
    pub payload: GgsMessage,
    pub signature: SignatureBundle,
    pub staking_score: f32,
}

pub struct ConsensusConfig {
    pub heartbeat_timeout: Duration,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout: Duration::from_secs(300),
        }
    }
}

pub struct ConsensusEngine {
    crypto: Arc<CryptoSuite>,
    ledger: RwLock<HashMap<String, StakeRecord>>,
    config: ConsensusConfig,
}

impl ConsensusEngine {
    pub fn new(crypto: Arc<CryptoSuite>, config: ConsensusConfig) -> Self {
        Self {
            crypto,
            ledger: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub fn sign(&self, payload: GgsMessage) -> anyhow::Result<SignedGossip> {
        let bytes = serde_json::to_vec(&payload)?;
        let signature = self.crypto.sign_bytes(&bytes)?;
        let peer_id = match &payload {
            GgsMessage::Heartbeat { peer, .. }
            | GgsMessage::SimilarityProbe { sender: peer, .. }
            | GgsMessage::SparseUpdate { sender: peer, .. }
            | GgsMessage::DenseSnapshot { sender: peer, .. } => peer.clone(),
        };
        let staking_score = self
            .ledger
            .read()
            .get(&peer_id)
            .map(|record| record.combined_weight())
            .unwrap_or(0.1);
        Ok(SignedGossip {
            payload,
            signature,
            staking_score,
        })
    }

    pub fn verify(&self, msg: &SignedGossip) -> bool {
        if let Ok(bytes) = serde_json::to_vec(&msg.payload) {
            return self.crypto.verify(&bytes, &msg.signature);
        }
        false
    }

    pub fn update_stake(&self, peer: &str, delta_eth: f64, delta_sol: f64, reputation_delta: f64) {
        let mut ledger = self.ledger.write();
        let entry = ledger.entry(peer.to_string()).or_insert(StakeRecord {
            stake_eth: 1.0,
            stake_sol: 0.1,
            reputation: 1.0,
            last_seen: Instant::now(),
        });
        entry.stake_eth = (entry.stake_eth + delta_eth).max(0.0);
        entry.stake_sol = (entry.stake_sol + delta_sol).max(0.0);
        entry.reputation = (entry.reputation + reputation_delta).max(-1.0);
        entry.last_seen = Instant::now();
    }

    pub fn prune_stale(&self) {
        let mut ledger = self.ledger.write();
        let deadline = Instant::now() - self.config.heartbeat_timeout;
        ledger.retain(|_, record| record.last_seen >= deadline);
    }

    pub fn stake_weight(&self, peer: &str) -> f32 {
        self.ledger
            .read()
            .get(peer)
            .map(|record| record.combined_weight())
            .unwrap_or(0.0)
    }
}
