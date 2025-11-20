use crate::types::GeoPoint;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const EMBEDDING_WEIGHT: f32 = 0.6;
const GEO_WEIGHT: f32 = 0.4;

#[derive(Clone, Debug)]
pub struct PeerProfile {
    pub embedding: Vec<f32>,
    pub position: GeoPoint,
    pub similarity: f32,
    pub geo_affinity: f32,
    pub score: f32,
    pub last_seen: Instant,
}

#[derive(Clone)]
pub struct TopologyConfig {
    pub max_neighbors: usize,
    pub failover_pool: usize,
    pub min_score: f32,
    pub geo_scale_km: f32,
    pub peer_stale_secs: u64,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 8,
            failover_pool: 4,
            min_score: 0.15,
            geo_scale_km: 500.0,
            peer_stale_secs: 120,
        }
    }
}

pub struct TopologySelector {
    position: GeoPoint,
    peers: RwLock<HashMap<String, PeerProfile>>,
    config: TopologyConfig,
}

#[derive(Debug, Clone)]
pub struct PeerSnapshot {
    pub similarity: f32,
    pub geo_affinity: f32,
    pub position: GeoPoint,
    pub embedding_dim: usize,
}

impl TopologySelector {
    pub fn new(position: GeoPoint, config: TopologyConfig) -> Self {
        Self {
            position,
            peers: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub fn position(&self) -> GeoPoint {
        self.position.clone()
    }

    pub fn update_peer(
        &self,
        peer_id: &str,
        embedding: Vec<f32>,
        position: GeoPoint,
        self_embedding: &[f32],
    ) {
        let similarity = cosine_sim(self_embedding, &embedding);
        let geo_affinity = self.geo_affinity(&position);
        let score = EMBEDDING_WEIGHT * similarity + GEO_WEIGHT * geo_affinity;
        let profile = PeerProfile {
            embedding,
            position,
            similarity,
            geo_affinity,
            score,
            last_seen: Instant::now(),
        };
        let mut peers = self.peers.write();
        peers.insert(peer_id.to_string(), profile);
        self.cleanup_locked(&mut peers);
    }

    pub fn neighbor_sets(&self) -> (Vec<String>, Vec<String>) {
        let peers = self.peers.read();
        let mut ranked: Vec<_> = peers.iter().collect();
        ranked.sort_by(|(_, a), (_, b)| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        let mut primary = Vec::new();
        let mut backups = Vec::new();
        for (_idx, (peer, profile)) in ranked.into_iter().enumerate() {
            if profile.score < self.config.min_score {
                continue;
            }
            if primary.len() < self.config.max_neighbors {
                primary.push(peer.clone());
            } else if backups.len() < self.config.failover_pool {
                backups.push(peer.clone());
            } else {
                break;
            }
        }
        (primary, backups)
    }

    pub fn select_neighbors(&self) -> Vec<String> {
        self.neighbor_sets().0
    }

    pub fn mark_unreachable(&self, peer_id: &str) {
        let mut peers = self.peers.write();
        peers.remove(peer_id);
    }

    pub fn max_neighbors(&self) -> usize {
        self.config.max_neighbors
    }

    pub fn failover_pool(&self) -> usize {
        self.config.failover_pool
    }

    pub fn peer_snapshot(&self, peer_id: &str) -> Option<PeerSnapshot> {
        self.peers.read().get(peer_id).map(|profile| PeerSnapshot {
            similarity: profile.similarity,
            geo_affinity: profile.geo_affinity,
            position: profile.position.clone(),
            embedding_dim: profile.embedding.len(),
        })
    }

    pub fn geo_affinity(&self, other: &GeoPoint) -> f32 {
        let dist = self.position.distance_km(other);
        (self.config.geo_scale_km / (self.config.geo_scale_km + dist)).clamp(0.0, 1.0)
    }

    fn cleanup_locked(&self, peers: &mut HashMap<String, PeerProfile>) {
        let deadline = Instant::now() - Duration::from_secs(self.config.peer_stale_secs);
        peers.retain(|_, profile| profile.last_seen >= deadline);
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
