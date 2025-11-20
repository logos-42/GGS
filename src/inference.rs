use crate::types::{decompress_indices, SparseUpdate, TensorSnapshot};
use anyhow::{anyhow, Result};
use ndarray::Array1;
use ndarray_npy::ReadNpyExt;
use parking_lot::RwLock;
use rand::Rng;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct InferenceConfig {
    pub model_dim: usize,
    pub model_path: Option<PathBuf>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_dim: 256,
            model_path: None,
        }
    }
}

#[derive(Clone)]
pub struct InferenceEngine {
    state: Arc<RwLock<ModelState>>,
    config: InferenceConfig,
}

struct ModelState {
    params: Array1<f32>,
    residual: Array1<f32>,
    version: u64,
}

impl InferenceEngine {
    pub fn new(config: InferenceConfig) -> Result<Self> {
        let params = load_or_random(config.model_dim, config.model_path.as_deref())?;
        let residual = Array1::<f32>::zeros(params.len());
        Ok(Self {
            state: Arc::new(RwLock::new(ModelState {
                params,
                residual,
                version: 1,
            })),
            config,
        })
    }

    pub fn model_dim(&self) -> usize {
        self.config.model_dim
    }

    pub fn embedding(&self) -> Vec<f32> {
        self.state.read().params.to_vec()
    }

    pub fn tensor_snapshot(&self) -> TensorSnapshot {
        let state = self.state.read();
        TensorSnapshot::new(state.params.to_vec(), state.version)
    }

    pub fn tensor_hash(&self) -> String {
        self.tensor_snapshot().hash()
    }

    pub fn make_sparse_update(&self, k: usize) -> SparseUpdate {
        let mut state = self.state.write();
        let dim = state.params.len();
        if dim == 0 {
            return SparseUpdate {
                indices: Vec::new(),
                values: Vec::new(),
                version: state.version,
            };
        }
        let mut delta = vec![0f32; dim];
        for i in 0..dim {
            delta[i] = state.params[i] + state.residual[i];
        }
        let mut idx_val: Vec<(usize, f32)> =
            delta.iter().enumerate().map(|(i, v)| (i, *v)).collect();
        idx_val.sort_by(|a, b| {
            let av = a.1.abs();
            let bv = b.1.abs();
            bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
        });
        let take = k.min(dim);
        let topk = &idx_val[..take];
        let mut sparse_vals = Vec::with_capacity(take);
        let mut sparse_idx = Vec::with_capacity(take);
        let mut last = 0usize;
        for (i, v) in topk {
            let diff = if sparse_idx.is_empty() {
                *i as u32
            } else {
                (*i - last) as u32
            };
            sparse_idx.push(diff);
            sparse_vals.push(*v);
            state.residual[*i] = delta[*i] - *v;
            last = *i;
        }
        state.version = state.version.saturating_add(1);
        SparseUpdate {
            indices: sparse_idx,
            values: sparse_vals,
            version: state.version,
        }
    }

    pub fn apply_sparse_update(&self, update: &SparseUpdate) {
        if update.indices.is_empty() {
            return;
        }
        let idxs = decompress_indices(&update.indices);
        let mut state = self.state.write();
        for (pos, &v) in idxs.iter().zip(update.values.iter()) {
            if *pos < state.params.len() {
                let old = state.params[*pos];
                let merged = 0.5 * old + 0.5 * v;
                state.params[*pos] = merged;
                state.residual[*pos] += old - merged;
            }
        }
        state.version = state.version.max(update.version);
    }

    pub fn apply_dense_snapshot(&self, snapshot: &TensorSnapshot) {
        let mut state = self.state.write();
        let len = state.params.len().min(snapshot.values.len());
        for i in 0..len {
            state.params[i] = 0.8 * state.params[i] + 0.2 * snapshot.values[i];
        }
        state.version = state.version.max(snapshot.version);
    }

    pub fn local_train_step(&self) {
        let mut rng = rand::thread_rng();
        let mut state = self.state.write();
        for v in state.params.iter_mut() {
            *v += rng.gen_range(-1e-3..1e-3);
        }
        state.version = state.version.saturating_add(1);
    }
}

fn load_or_random(dim: usize, path: Option<&Path>) -> Result<Array1<f32>> {
    if let Some(path) = path {
        if path.exists() {
            let file = File::open(path)?;
            let arr: Array1<f32> = Array1::read_npy(file)?;
            return Ok(arr);
        } else {
            return Err(anyhow!("model file {:?} not found", path));
        }
    }
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
    Ok(Array1::from_vec(data))
}
