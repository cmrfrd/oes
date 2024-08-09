use crate::DEFAULT_TOPIC_CHANNEL_SIZE;
use anyhow::Error;
use dashmap::DashMap;
use flume::r#async::RecvStream;
use flume::{bounded, SendError};
use flume::{Receiver, Sender};
use futures::future::join_all;
use futures::SinkExt;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone)]
pub struct Topic<T> {
    pub tx: Sender<T>,
    pub rx: Receiver<T>,
    created_at: std::time::Instant,
}

impl<T> Topic<T> {
    pub fn new(txrx: (Sender<T>, Receiver<T>)) -> Self {
        Self {
            tx: txrx.0,
            rx: txrx.1,
            created_at: std::time::Instant::now(),
        }
    }
}

#[derive(Clone)]
pub struct Broker<T> {
    pub topics: Arc<DashMap<String, Topic<T>>>,
}

pub struct Stats {
    pub num_topics: usize,
}

impl<T> Default for Broker<T> {
    fn default() -> Self {
        Self {
            topics: Arc::new(DashMap::new()),
        }
    }
}

impl<T> AsRef<Broker<T>> for Broker<T> {
    fn as_ref(&self) -> &Broker<T> {
        self
    }
}

impl<T: Debug + Sync + Send + 'static> Broker<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

pub trait PubSub<T: Debug + Sync + Send + 'static> {
    fn stats(&self) -> Stats;
    fn has_topic(&self, topic: &str) -> bool;
    fn subscribe(&mut self, topic: String) -> RecvStream<T>;
    fn unsubscribe(&mut self, topic: String);
    fn try_recv_many(&mut self, topic: String, n: usize) -> Vec<T>;
    fn publish_single(
        &mut self,
        topic: String,
        messages: Vec<T>,
    ) -> impl std::future::Future<Output = Result<(), SendError<T>>> + Send;
    fn publish_many(
        &mut self,
        messages: Vec<(String, T)>,
        only_if_topic_exists: bool,
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;
}

impl<T: Debug + Sync + Send + 'static> PubSub<T> for Broker<T> {
    fn stats(&self) -> Stats {
        let num_topics = self.topics.len();
        Stats { num_topics }
    }

    fn has_topic(&self, topic: &str) -> bool {
        self.topics.contains_key(topic)
    }

    fn subscribe(&mut self, topic: String) -> RecvStream<T> {
        self.topics
            .entry(topic)
            .or_insert_with(|| Topic::new(bounded(DEFAULT_TOPIC_CHANNEL_SIZE)))
            .value()
            .rx
            .clone()
            .into_stream()
    }

    fn unsubscribe(&mut self, topic: String) {
        self.topics.remove(&topic);
    }

    fn try_recv_many(&mut self, topic: String, n: usize) -> Vec<T> {
        let rx = self
            .topics
            .entry(topic)
            .or_insert_with(|| Topic::new(bounded(DEFAULT_TOPIC_CHANNEL_SIZE)))
            .value()
            .rx
            .clone();

        let mut messages = Vec::with_capacity(n);
        for _ in 0..n {
            match rx.try_recv() {
                Ok(msg) => messages.push(msg),
                Err(_) => break,
            }
        }
        messages
    }

    async fn publish_single(
        &mut self,
        topic: String,
        messages: Vec<T>,
    ) -> Result<(), flume::SendError<T>> {
        self.topics
            .entry(topic)
            .or_insert_with(|| Topic::new(bounded(DEFAULT_TOPIC_CHANNEL_SIZE)))
            .value()
            .tx
            .clone()
            .into_sink()
            .send_all(&mut futures::stream::iter(messages.into_iter().map(Ok)))
            .await
    }

    async fn publish_many(
        &mut self,
        messages: Vec<(String, T)>,
        only_if_topic_exists: bool,
    ) -> Result<(), anyhow::Error> {
        let futs = messages
            .into_iter()
            .filter(|(topic, _)| {
                if only_if_topic_exists {
                    self.topics.contains_key(topic)
                } else {
                    true
                }
            })
            .map(|(topic, message)| {
                let tx = self
                    .topics
                    .entry(topic.to_string())
                    .or_insert_with(|| Topic::new(bounded(DEFAULT_TOPIC_CHANNEL_SIZE)))
                    .value()
                    .tx
                    .clone();
                async move { tx.send_async(message).await }
            })
            .collect::<Vec<_>>();
        Ok(for res in join_all(futs.into_iter()).await {
            match res {
                Ok(_) => {}
                Err(e) => return Err(e.into()),
            }
        })
    }
}
