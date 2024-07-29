use crate::DEFAULT_TOPIC_CHANNEL_SIZE;
use anyhow::Error;
use dashmap::DashMap;
use flume::r#async::RecvStream;
use flume::{bounded, SendError};
use flume::{Receiver, Sender};
use futures::future::join_all;
use futures::SinkExt;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

#[derive(Clone)]
pub struct Broker<T: Clone> {
    topics: Arc<DashMap<String, (Sender<T>, Receiver<T>)>>,
}

impl<T: Clone> Default for Broker<T> {
    fn default() -> Self {
        Self {
            topics: Arc::new(DashMap::new()),
        }
    }
}

impl<T: Clone> AsRef<Broker<T>> for Broker<T> {
    fn as_ref(&self) -> &Broker<T> {
        self
    }
}

impl<T: Clone + Debug + Unpin + Sync + Send + 'static> Broker<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn current_sizes(&mut self) -> HashMap<String, usize> {
        let mut capacities = HashMap::new();
        for ee in self.topics.iter() {
            let (topic, (_, rx)) = ee.pair();
            capacities.insert(topic.clone(), rx.len());
        }
        capacities
    }
}
pub trait PubSub<T: Clone + Debug + Unpin + Sync + Send + 'static> {
    fn subscribe(&mut self, topic: String) -> RecvStream<T>;
    fn unsubscribe(&mut self, topic: String);
    fn try_recv_many(
        &mut self,
        topic: String,
        n: usize,
    ) -> impl std::future::Future<Output = Vec<T>> + Send;
    fn publish_single(
        &mut self,
        topic: String,
        messages: Vec<T>,
    ) -> impl std::future::Future<Output = Result<(), SendError<T>>> + Send;
    fn publish_many(
        &mut self,
        messages: Vec<(String, T)>,
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;
}

impl<T: Clone + Debug + Unpin + Sync + Send + 'static> PubSub<T> for Broker<T> {
    fn subscribe(&mut self, topic: String) -> RecvStream<T> {
        self.topics
            .entry(topic)
            .or_insert_with(|| bounded(DEFAULT_TOPIC_CHANNEL_SIZE))
            .value()
            .1
            .clone()
            .into_stream()
    }

    async fn try_recv_many(&mut self, topic: String, n: usize) -> Vec<T> {
        let rx = self
            .topics
            .entry(topic)
            .or_insert_with(|| bounded(DEFAULT_TOPIC_CHANNEL_SIZE))
            .value()
            .1
            .clone();

        let mut messages = Vec::with_capacity(n);
        for _ in 0..n {
            match rx.try_recv() {
                Ok(msg) => messages.push(msg),
                Err(_) => break,
            }
            tokio::task::yield_now().await;
        }
        messages
    }

    fn unsubscribe(&mut self, topic: String) {
        self.topics.remove(&topic);
    }

    async fn publish_single(
        &mut self,
        topic: String,
        messages: Vec<T>,
    ) -> Result<(), flume::SendError<T>> {
        self.topics
            .entry(topic)
            .or_insert_with(|| bounded(DEFAULT_TOPIC_CHANNEL_SIZE))
            .value()
            .0
            .clone()
            .into_sink()
            .send_all(&mut futures::stream::iter(messages.into_iter().map(Ok)))
            .await
    }

    async fn publish_many(&mut self, messages: Vec<(String, T)>) -> Result<(), anyhow::Error> {
        let mut futs = Vec::with_capacity(messages.len());
        for (topic, message) in messages {
            let tx = self
                .topics
                .entry(topic)
                .or_insert_with(|| bounded(DEFAULT_TOPIC_CHANNEL_SIZE))
                .value()
                .0
                .clone();
            futs.push(tokio::spawn(async move { tx.send_async(message).await }));
        }
        Ok(for res in join_all(futs.into_iter()).await {
            match res {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(e.into()),
                Err(e) => return Err(e.into()),
            }
        })
    }
}
