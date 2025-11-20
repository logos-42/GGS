use crate::consensus::SignedGossip;
use anyhow::{anyhow, Result};
use libp2p::{
    gossipsub::{
        self, Behaviour as GossipsubBehaviour, Event as GossipsubEvent, IdentTopic as Topic,
        MessageAuthenticity, ValidationMode,
    },
    identity,
    mdns::{self, tokio::Behaviour as Mdns, Event as MdnsEvent},
    swarm::{NetworkBehaviour, SwarmBuilder},
    Multiaddr, PeerId, Swarm,
};
use parking_lot::RwLock;
use quinn::{Endpoint, ServerConfig};
use rcgen::generate_simple_self_signed;
use rustls::{Certificate, PrivateKey};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct CommsConfig {
    pub topic: String,
    pub listen_addr: Option<Multiaddr>,
    pub quic_bind: Option<SocketAddr>,
    pub quic_bootstrap: Vec<SocketAddr>,
    pub bandwidth: BandwidthBudgetConfig,
}

impl Default for CommsConfig {
    fn default() -> Self {
        Self {
            topic: "ggs-training".into(),
            listen_addr: None,
            quic_bind: Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9234)),
            quic_bootstrap: Vec::new(),
            bandwidth: BandwidthBudgetConfig::default(),
        }
    }
}

#[derive(Clone)]
pub struct BandwidthBudgetConfig {
    pub sparse_per_window: u32,
    pub dense_bytes_per_window: usize,
    pub window_secs: u64,
}

impl Default for BandwidthBudgetConfig {
    fn default() -> Self {
        Self {
            sparse_per_window: 12,
            dense_bytes_per_window: 256 * 1024,
            window_secs: 60,
        }
    }
}

struct BandwidthBudget {
    config: BandwidthBudgetConfig,
    window_start: Instant,
    sparse_sent: u32,
    dense_sent: usize,
}

impl BandwidthBudget {
    fn new(config: BandwidthBudgetConfig) -> Self {
        Self {
            config,
            window_start: Instant::now(),
            sparse_sent: 0,
            dense_sent: 0,
        }
    }

    fn rotate(&mut self) {
        if self.window_start.elapsed() >= Duration::from_secs(self.config.window_secs) {
            self.window_start = Instant::now();
            self.sparse_sent = 0;
            self.dense_sent = 0;
        }
    }

    fn allow_sparse(&mut self) -> bool {
        self.rotate();
        if self.sparse_sent < self.config.sparse_per_window {
            self.sparse_sent += 1;
            true
        } else {
            false
        }
    }

    fn allow_dense(&mut self, bytes: usize) -> bool {
        self.rotate();
        if self.dense_sent + bytes <= self.config.dense_bytes_per_window {
            self.dense_sent += bytes;
            true
        } else {
            false
        }
    }
}

#[derive(NetworkBehaviour)]
#[behaviour(out_event = "OutEvent")]
pub struct Behaviour {
    gossipsub: GossipsubBehaviour,
    mdns: Mdns,
}

#[derive(Debug)]
pub enum OutEvent {
    Gossipsub(GossipsubEvent),
    Mdns(MdnsEvent),
}

impl From<GossipsubEvent> for OutEvent {
    fn from(v: GossipsubEvent) -> Self {
        OutEvent::Gossipsub(v)
    }
}

impl From<MdnsEvent> for OutEvent {
    fn from(v: MdnsEvent) -> Self {
        OutEvent::Mdns(v)
    }
}

pub struct CommsHandle {
    pub peer_id: PeerId,
    pub swarm: Swarm<Behaviour>,
    pub topic: Topic,
    quic: Option<Arc<QuicGateway>>,
    bandwidth: RwLock<BandwidthBudget>,
}

impl CommsHandle {
    pub async fn new(config: CommsConfig) -> Result<Self> {
        let local_key = identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(local_key.public());

        let transport = libp2p::tokio_development_transport(local_key.clone())?;
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .validation_mode(ValidationMode::Permissive)
            .build()
            .expect("valid config");
        let mut gossipsub = GossipsubBehaviour::new(
            MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        )
        .map_err(|e| anyhow!(e))?;
        let topic = Topic::new(config.topic.clone());
        gossipsub.subscribe(&topic)?;
        let mdns = Mdns::new(mdns::Config::default(), peer_id)?;
        let behaviour = Behaviour { gossipsub, mdns };
        let mut swarm = SwarmBuilder::with_tokio_executor(transport, behaviour, peer_id).build();
        if let Some(addr) = config.listen_addr {
            swarm.listen_on(addr)?;
        }

        let quic = if let Some(bind) = config.quic_bind {
            let gateway = Arc::new(QuicGateway::new(bind)?);
            for addr in &config.quic_bootstrap {
                let _ = gateway.connect(*addr).await;
            }
            Some(gateway)
        } else {
            None
        };

        Ok(Self {
            peer_id: swarm.local_peer_id().clone(),
            swarm,
            topic,
            quic,
            bandwidth: RwLock::new(BandwidthBudget::new(config.bandwidth)),
        })
    }

    pub fn publish(&mut self, signed: &SignedGossip) -> Result<()> {
        let data = serde_json::to_vec(signed)?;
        self.swarm
            .behaviour_mut()
            .gossipsub
            .publish(self.topic.clone(), data)?;
        Ok(())
    }

    pub fn allow_sparse_update(&self) -> bool {
        self.bandwidth.write().allow_sparse()
    }

    pub fn allow_dense_snapshot(&self, bytes: usize) -> bool {
        self.bandwidth.write().allow_dense(bytes)
    }

    pub async fn broadcast_realtime(&self, signed: &SignedGossip) -> bool {
        if let Some(quic) = &self.quic {
            return quic.broadcast(signed).await;
        }
        false
    }
}

struct QuicGateway {
    endpoint: Endpoint,
    connections: Arc<RwLock<Vec<quinn::Connection>>>,
}

impl QuicGateway {
    fn new(bind: SocketAddr) -> Result<Self> {
        let cert = generate_simple_self_signed(vec!["ggs-quic".into()])?;
        let cert_der = cert.serialize_der()?;
        let key_der = cert.serialize_private_key_der();
        let mut server_config = ServerConfig::with_single_cert(
            vec![Certificate(cert_der.clone())],
            PrivateKey(key_der.clone()),
        )?;
        server_config.transport = Arc::new(quinn::TransportConfig::default());
        let endpoint = Endpoint::server(server_config, bind)?;
        let connections = Arc::new(RwLock::new(Vec::new()));
        let accept_endpoint = endpoint.clone();
        let accept_pool = connections.clone();
        tokio::spawn(async move {
            loop {
                match accept_endpoint.accept().await {
                    Some(connecting) => match connecting.await {
                        Ok(conn) => accept_pool.write().push(conn),
                        Err(err) => eprintln!("[QUIC] accept error: {err:?}"),
                    },
                    None => tokio::time::sleep(Duration::from_secs(1)).await,
                }
            }
        });
        Ok(Self {
            endpoint,
            connections,
        })
    }

    async fn connect(&self, addr: SocketAddr) -> Result<()> {
        match self.endpoint.connect(addr, "ggs-quic") {
            Ok(connecting) => match connecting.await {
                Ok(connection) => {
                    self.connections.write().push(connection);
                    Ok(())
                }
                Err(err) => Err(err.into()),
            },
            Err(err) => Err(err.into()),
        }
    }

    async fn broadcast(&self, signed: &SignedGossip) -> bool {
        let bytes = match serde_json::to_vec(signed) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let entries: Vec<(usize, quinn::Connection)> = {
            let guard = self.connections.read();
            guard.iter().cloned().enumerate().collect()
        };
        let mut success = false;
        let mut dead_indices = Vec::new();
        for (idx, conn) in entries {
            match conn.open_uni().await {
                Ok(mut send) => {
                    if send.write_all(&bytes).await.is_ok() && send.finish().await.is_ok() {
                        success = true;
                    }
                }
                Err(_) => dead_indices.push(idx),
            }
        }
        if !dead_indices.is_empty() {
            let mut guard = self.connections.write();
            for idx in dead_indices.into_iter().rev() {
                let _ = guard.swap_remove(idx);
            }
        }
        success
    }
}
