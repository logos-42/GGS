use rand::Rng;
use serde::{Deserialize, Serialize};

/// 地理位置点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f32,
    pub lon: f32,
}

impl GeoPoint {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            lat: rng.gen_range(-60.0..60.0),
            lon: rng.gen_range(-180.0..180.0),
        }
    }

    pub fn distance_km(&self, other: &Self) -> f32 {
        const EARTH_RADIUS_KM: f32 = 6_371.0;
        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        EARTH_RADIUS_KM * c
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSnapshot {
    pub dim: usize,
    pub values: Vec<f32>,
    pub version: u64,
}

impl TensorSnapshot {
    pub fn new(values: Vec<f32>, version: u64) -> Self {
        Self {
            dim: values.len(),
            values,
            version,
        }
    }

    pub fn hash(&self) -> String {
        use sha3::{Digest, Keccak256};
        let mut hasher = Keccak256::new();
        hasher.update(self.dim.to_le_bytes());
        hasher.update(self.version.to_le_bytes());
        for v in &self.values {
            hasher.update(v.to_ne_bytes());
        }
        format!("0x{}", hex::encode(hasher.finalize()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseUpdate {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub version: u64,
}

pub fn decompress_indices(compressed: &[u32]) -> Vec<usize> {
    let mut out = Vec::with_capacity(compressed.len());
    let mut last = 0usize;
    for diff in compressed {
        let next = last + (*diff as usize);
        out.push(next);
        last = next;
    }
    out
}

/// Gossip 消息体
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GgsMessage {
    Heartbeat {
        peer: String,
        model_hash: String,
    },
    SparseUpdate {
        update: SparseUpdate,
        sender: String,
    },
    DenseSnapshot {
        snapshot: TensorSnapshot,
        sender: String,
    },
    SimilarityProbe {
        embedding: Vec<f32>,
        position: GeoPoint,
        sender: String,
    },
}
