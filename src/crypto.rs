use anyhow::{anyhow, Result};
use ed25519_dalek::{
    Keypair as SolKeypair, PublicKey as SolPublicKey, SecretKey as SolSecretKey,
    Signature as SolRawSignature, Signer as SolSigner, Verifier as SolVerifier,
};
use k256::ecdsa::{
    signature::{Signer, Verifier},
    Signature as EthSignatureRaw, SigningKey, VerifyingKey,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::convert::TryInto;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EthSignature {
    pub address: String,
    pub signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolSignature {
    pub pubkey: String,
    pub signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignatureBundle {
    pub eth: EthSignature,
    pub sol: SolSignature,
}

pub struct CryptoConfig {
    pub eth_hex_seed: Option<String>,
    pub sol_bs58_seed: Option<String>,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            eth_hex_seed: None,
            sol_bs58_seed: None,
        }
    }
}

#[derive(Clone)]
pub struct CryptoSuite {
    eth: Arc<EthIdentity>,
    sol: Arc<SolIdentity>,
}

impl CryptoSuite {
    pub fn new(config: CryptoConfig) -> Result<Self> {
        let eth = EthIdentity::new(config.eth_hex_seed)?;
        let sol = SolIdentity::new(config.sol_bs58_seed)?;
        Ok(Self {
            eth: Arc::new(eth),
            sol: Arc::new(sol),
        })
    }

    pub fn sign_bytes(&self, payload: &[u8]) -> Result<SignatureBundle> {
        let eth_sig = self.eth.sign(payload)?;
        let sol_sig = self.sol.sign(payload)?;
        Ok(SignatureBundle {
            eth: eth_sig,
            sol: sol_sig,
        })
    }

    pub fn verify(&self, payload: &[u8], sig: &SignatureBundle) -> bool {
        self.eth.verify(payload, &sig.eth) && self.sol.verify(payload, &sig.sol)
    }

    pub fn eth_address(&self) -> String {
        self.eth.address.clone()
    }

    pub fn sol_address(&self) -> String {
        self.sol.pubkey.clone()
    }
}

struct EthIdentity {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    address: String,
}

impl EthIdentity {
    fn new(seed: Option<String>) -> Result<Self> {
        let secret = if let Some(seed_hex) = seed {
            let bytes = hex::decode(seed_hex.trim_start_matches("0x"))?;
            let arr: [u8; 32] = bytes
                .try_into()
                .map_err(|_| anyhow!("eth seed must be 32 bytes"))?;
            arr
        } else {
            random_bytes()
        };
        let signing_key =
            SigningKey::from_bytes(&secret.into()).map_err(|e| anyhow!(e.to_string()))?;
        let verifying_key = signing_key.verifying_key().clone();
        let address = eth_address_from_key(&verifying_key);
        Ok(Self {
            signing_key,
            verifying_key,
            address,
        })
    }

    fn sign(&self, payload: &[u8]) -> Result<EthSignature> {
        let digest = keccak(payload);
        let signature: EthSignatureRaw = self.signing_key.sign(&digest);
        Ok(EthSignature {
            address: self.address.clone(),
            signature: hex::encode(signature.to_vec()),
        })
    }

    fn verify(&self, payload: &[u8], sig: &EthSignature) -> bool {
        if sig.address.to_lowercase() != self.address.to_lowercase() {
            return false;
        }
        if let Ok(bytes) = hex::decode(&sig.signature) {
            if let Ok(signature) = EthSignatureRaw::try_from(bytes.as_slice()) {
                let digest = keccak(payload);
                return self.verifying_key.verify(&digest, &signature).is_ok();
            }
        }
        false
    }
}

struct SolIdentity {
    keypair: SolKeypair,
    pubkey: String,
}

impl SolIdentity {
    fn new(seed: Option<String>) -> Result<Self> {
        let keypair = if let Some(bs58_seed) = seed {
            let bytes = bs58::decode(bs58_seed).into_vec()?;
            match bytes.len() {
                32 => {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    keypair_from_secret(arr)?
                }
                64 => {
                    let mut arr = [0u8; 64];
                    arr.copy_from_slice(&bytes);
                    SolKeypair::from_bytes(&arr).map_err(|e| anyhow!("sol key error: {e}"))?
                }
                _ => return Err(anyhow!("Solana seed must be 32 or 64 bytes")),
            }
        } else {
            keypair_from_secret(random_bytes())?
        };
        let pubkey = bs58::encode(keypair.public.as_bytes()).into_string();
        Ok(Self { keypair, pubkey })
    }

    fn sign(&self, payload: &[u8]) -> Result<SolSignature> {
        let signature = self.keypair.sign(payload);
        Ok(SolSignature {
            pubkey: self.pubkey.clone(),
            signature: bs58::encode(signature.to_bytes()).into_string(),
        })
    }

    fn verify(&self, payload: &[u8], sig: &SolSignature) -> bool {
        if sig.pubkey != self.pubkey {
            return false;
        }
        if let Ok(bytes) = bs58::decode(&sig.signature).into_vec() {
            if let Ok(signature) = SolRawSignature::from_bytes(&bytes) {
                return self.keypair.public.verify(payload, &signature).is_ok();
            }
        }
        false
    }
}

fn eth_address_from_key(key: &VerifyingKey) -> String {
    let encoded = key.to_encoded_point(false);
    let public_key = encoded.as_bytes();
    // drop first byte 0x04
    let hash = keccak(&public_key[1..]);
    format!("0x{}", hex::encode(&hash[12..]))
}

fn keccak(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(payload);
    hasher.finalize().into()
}

fn random_bytes() -> [u8; 32] {
    let mut buf = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut buf);
    buf
}

fn keypair_from_secret(secret_bytes: [u8; 32]) -> Result<SolKeypair> {
    let secret =
        SolSecretKey::from_bytes(&secret_bytes).map_err(|e| anyhow!("sol key error: {e}"))?;
    let public = SolPublicKey::from(&secret);
    Ok(SolKeypair { secret, public })
}
