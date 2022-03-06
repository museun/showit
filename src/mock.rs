use std::{
    borrow::Cow,
    io::Read,
    sync::{atomic::Ordering, Arc},
};

use reqwest::blocking::Body;

use crate::{DataProgress, Progress, UploadResult, Uploader};

mod limit {
    use std::time::{Duration, Instant};

    pub struct RateLimit {
        cap: u64,
        bucket: Bucket,
    }

    impl RateLimit {
        pub fn new(cap: u64, initial: u64, period: Duration) -> Self {
            Self {
                cap,
                bucket: Bucket::new(cap, initial, period),
            }
        }

        pub fn full(cap: u64, period: Duration) -> Self {
            Self {
                cap,
                bucket: Bucket::new(cap, cap, period),
            }
        }

        pub fn empty(cap: u64, period: Duration) -> Self {
            Self {
                cap,
                bucket: Bucket::new(cap, 0, period),
            }
        }

        pub fn consume(&mut self, tokens: u64) -> Result<u64, Duration> {
            let now = Instant::now();
            let mut bucket = &mut self.bucket;
            if let Some(n) = bucket.refill(now) {
                bucket.tokens = std::cmp::min(bucket.tokens + n, self.cap);
            };

            if tokens <= bucket.tokens {
                bucket.tokens -= tokens;
                bucket.backoff = 0;
                return Ok(bucket.tokens);
            }

            let prev = bucket.tokens;
            Err(bucket.estimate(tokens - prev, now))
        }

        pub fn throttle(&mut self, tokens: u64) -> u64 {
            loop {
                match self.consume(tokens) {
                    Ok(rem) => break rem,
                    Err(dt) => std::thread::sleep(dt),
                }
            }
        }

        pub fn take(&mut self) -> u64 {
            self.throttle(1)
        }
    }

    struct Bucket {
        tokens: u64,
        backoff: u32,
        next: Instant,
        last: Instant,
        quantum: u64,
        period: Duration,
    }

    impl Bucket {
        fn new(tokens: u64, initial: u64, period: Duration) -> Self {
            let now = Instant::now();
            Self {
                tokens: initial,
                backoff: 0,
                next: now + period,
                last: now,
                quantum: tokens,
                period,
            }
        }

        fn refill(&mut self, now: Instant) -> Option<u64> {
            if now < self.next {
                return None;
            }

            let last = now.duration_since(self.last);
            let periods = last.as_nanos().checked_div(self.period.as_nanos())? as u64;
            self.last += self.period * (periods as u32);
            self.next = self.last + self.period;
            Some(periods * self.quantum)
        }

        fn estimate(&mut self, tokens: u64, now: Instant) -> Duration {
            let until = self.next.duration_since(now);
            let periods = (tokens.checked_add(self.quantum).unwrap() - 1) / self.quantum;
            until + self.period * (periods as u32 - 1)
        }
    }
}

pub(crate) fn start_test_server() {
    fn slow_read(read: impl Read) {
        struct SlowRead<R>(R, limit::RateLimit);
        impl<R: Read> Read for SlowRead<R> {
            fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
                let n = self.0.read(buf)?;
                self.1.take();
                Ok(n)
            }
        }

        let _ = std::io::copy(
            &mut SlowRead(
                read,
                limit::RateLimit::empty(10, std::time::Duration::from_millis(10)),
            ),
            &mut std::io::sink(),
        );
    }

    #[derive(serde::Serialize)]
    struct ImageResp {
        url: String,
    }

    let id = || {
        std::iter::repeat_with(fastrand::alphanumeric)
            .take(5)
            .collect::<String>()
    };

    let server = Arc::new(tiny_http::Server::http("localhost:8888").unwrap());
    for _ in 0..4 {
        let _ = std::thread::spawn({
            let server = server.clone();
            move || {
                for mut req in server
                    .incoming_requests()
                    .filter(|req| matches!(req.method(), tiny_http::Method::Post))
                {
                    slow_read(req.as_reader());
                    let resp = ImageResp {
                        url: format!("http://localhost:8888/i/{}", id()),
                    };

                    let resp =
                        tiny_http::Response::from_string(serde_json::to_string(&resp).unwrap())
                            .with_header(
                                "Content-Type: application/json"
                                    .parse::<tiny_http::Header>()
                                    .unwrap(),
                            )
                            .with_status_code(200);

                    let _ = req.respond(resp);
                }
            }
        });
    }
}

#[derive(Debug)]
pub struct TestUploadResult {
    link: Box<str>,
}

impl UploadResult for TestUploadResult {
    fn link(&self) -> std::borrow::Cow<'_, str> {
        Cow::Borrowed(&*self.link)
    }

    fn delete_link(&self) -> Option<std::borrow::Cow<'_, str>> {
        None
    }
}

#[derive(Default, Clone)]
pub struct TestUploader;

impl Uploader for TestUploader {
    type UploadResult = TestUploadResult;

    fn upload(
        &mut self,
        data: Vec<u8>,
        name: String,
        progress: DataProgress,
    ) -> anyhow::Result<Self::UploadResult> {
        struct Payload(usize);
        impl Read for Payload {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = std::cmp::min(self.0, buf.len());
                self.0 -= n;
                Ok(n)
            }
        }

        let resp = reqwest::blocking::Client::new()
            .post("http://localhost:8888")
            .body(Body::new(Progress(Payload(data.len()), {
                move |d| {
                    progress.amount.fetch_add(d, Ordering::SeqCst);
                    progress.repaint.request_repaint();
                }
            })))
            .send()?;

        #[derive(serde::Deserialize)]
        struct TestResp {
            url: String,
        }

        resp.json()
            .map(|TestResp { url }| url.into_boxed_str())
            .map(|link| Self::UploadResult { link })
            .map_err(Into::into)
    }
}
