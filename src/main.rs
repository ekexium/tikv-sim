use hdrhistogram::Histogram;
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;

use rand::prelude::IteratorRandom;
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal};

type Events = Rc<RefCell<EventHeap>>;

type Time = u64;

trait TimeTrait {
    fn pretty_print(&self) -> String;
}

impl TimeTrait for Time {
    fn pretty_print(&self) -> String {
        if *self < MICROSECOND {
            format!("{} ns", self)
        } else if *self < MILLISECOND {
            format!("{:.2} us", *self as f64 / MICROSECOND as f64)
        } else if *self < SECOND {
            format!("{:.2} ms", *self as f64 / MILLISECOND as f64)
        } else {
            format!("{:.2} s", *self as f64 / SECOND as f64)
        }
    }
}

const NANOSECOND: Time = 1;
const MICROSECOND: Time = 1_000 * NANOSECOND;
const MILLISECOND: Time = 1_000 * MICROSECOND;
const SECOND: Time = 1_000 * MILLISECOND;

static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);
static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);
static SERVER_COUNTER: AtomicU64 = AtomicU64::new(0);
static CLIENT_COUNTER: AtomicU64 = AtomicU64::new(0);
static CURRENT_TIME: AtomicU64 = AtomicU64::new(0);

// https://aws.amazon.com/blogs/architecture/improving-performance-and-reducing-cost-using-availability-zone-affinity/
// https://www.xkyle.com/Measuring-AWS-Region-and-AZ-Latency/
lazy_static! {
    static ref REMOTE_NET_LATENCY_DIST: LogNormal<f64> = {
        let mean = 0.5 * MILLISECOND as f64;
        let std_dev = 0.1 * MILLISECOND as f64;
        let location = (mean.powi(2) / (mean.powi(2) + std_dev.powi(2)).sqrt()).ln();
        let scale = (1.0 + std_dev.powi(2) / mean.powi(2)).ln().sqrt();
        LogNormal::new(location, scale).unwrap()
    };
    static ref LOCAL_NET_LATENCY_DIST: LogNormal<f64> = {
        let mean = 0.05 * MILLISECOND as f64;
        let std_dev = 0.01 * MILLISECOND as f64;
        let location = (mean.powi(2) / (mean.powi(2) + std_dev.powi(2)).sqrt()).ln();
        let scale = (1.0 + std_dev.powi(2) / mean.powi(2)).ln().sqrt();
        LogNormal::new(location, scale).unwrap()
    };
}

fn main() {
    let events: Rc<RefCell<EventHeap>> = Rc::new(RefCell::new(EventHeap::new()));

    let mut model = Model {
        replica_num: 3,
        events: events.clone(),
        servers: vec![
            Server::new(Zone::AZ1, events.clone(), 4, 20, 1 * SECOND),
            Server::new(Zone::AZ2, events.clone(), 4, 20, 1 * SECOND),
            Server::new(Zone::AZ3, events.clone(), 4, 20, 1 * SECOND),
        ],
        clients: vec![
            Client::new(Zone::AZ1, events.clone()),
            Client::new(Zone::AZ2, events.clone()),
            Client::new(Zone::AZ3, events.clone()),
        ],
        app: App::new(events.clone(), 5000.0, Some(15 * SECOND), 100),
    };
    model.init();
    let max_time = 60 * SECOND;

    loop {
        let mut events_mut = events.borrow_mut();
        if events_mut
            .peek()
            .map(|e| e.trigger_time > max_time)
            .unwrap_or(true)
        {
            break;
        }
        let event = events_mut.pop().unwrap();
        // time cannot go back
        assert!(now() <= event.trigger_time);
        CURRENT_TIME.store(event.trigger_time, atomic::Ordering::SeqCst);
        drop(events_mut);
        (event.f)(&mut model);
    }

    println!(
        "simulation time: {}, generated {} tasks, finished {} requests(including retry)",
        now().pretty_print(),
        TASK_COUNTER.load(atomic::Ordering::SeqCst),
        model
            .clients
            .iter()
            .map(|c| c.latency_stat.len())
            .sum::<u64>()
    );
    println!("{} application retries", model.app.retry_count);
    println!(
        "finished {} transactions, {} still in flight",
        model.app.txn_duration_stat.len(),
        model.app.pending_transactions.len()
    );

    println!(
        "txn duration mean: {}, p99: {}",
        (model.app.txn_duration_stat.mean() as u64).pretty_print(),
        (model.app.txn_duration_stat.value_at_quantile(0.99) as u64).pretty_print()
    );

    for client in model.clients {
        println!(
            "\nclient {:?} mean: {}, p99: {}",
            client.id,
            (client.latency_stat.mean() as u64).pretty_print(),
            (client.latency_stat.value_at_quantile(0.99) as u64).pretty_print(),
        );
        for error in client.error_latency_stat.keys() {
            println!(
                "error {:?} {:?}, mean: {}, p99: {}",
                error,
                client.error_latency_stat[error].len(),
                (client.error_latency_stat[error].mean() as u64).pretty_print(),
                (client.error_latency_stat[error].value_at_quantile(0.99) as u64).pretty_print(),
            );
        }
    }

    for server in model.servers {
        println!("\nserver {}", server.server_id);
        println!(
            "handled {} read requests, {} in queue, {} timeouts, schedule wait mean: {}, p99: {}",
            server.read_schedule_wait_stat.len(),
            server.read_task_queue.len(),
            server.error_count,
            (server.read_schedule_wait_stat.mean() as u64).pretty_print(),
            (server.read_schedule_wait_stat.value_at_quantile(0.99) as u64).pretty_print(),
        );
        println!(
            "handled {} write requests, {} in queue, schedule wait mean: {}, p99: {}",
            server.write_schedule_wait_stat.len(),
            server.write_task_queue.len(),
            (server.write_schedule_wait_stat.mean() as u64).pretty_print(),
            (server.write_schedule_wait_stat.value_at_quantile(0.99) as u64).pretty_print(),
        );
    }
}

struct Model {
    events: Events,
    servers: Vec<Server>,
    clients: Vec<Client>,
    replica_num: usize,
    app: App,
}

impl Model {
    fn init(&mut self) {
        // create regions
        assert!(self.servers.len() >= self.replica_num);
        let mut leader_idx = 0;
        for region_id in 0..self.app.region_num {
            // leader in server[leader_idx], `replica_num-1` followers in server[leader_idx + 1]..server[leader_idx+replica_num-1]
            let mut leader = Peer {
                role: Role::Leader,
                server_id: self.servers[leader_idx].server_id,
                region_id,
                resolved_ts: 0,
                safe_ts: 0,
                lock_cf: HashSet::new(),
                advance_interval: 5 * SECOND,
                broadcast_interval: 5 * SECOND,
            };
            leader.update_resolved_ts(self.events.clone());
            leader.broadcast_safe_ts(self.events.clone());
            self.servers[leader_idx].peers.insert(region_id, leader);
            for follow_id in 1..=self.replica_num - 1 {
                let follower_idx = (leader_idx + follow_id) % self.servers.len();
                let server_id = self.servers[follower_idx].server_id;
                self.servers[follower_idx].peers.insert(
                    region_id,
                    Peer {
                        role: Role::Follower,
                        server_id,
                        region_id,
                        resolved_ts: 0,
                        safe_ts: 0,
                        lock_cf: HashSet::new(),
                        advance_interval: 5 * SECOND,
                        broadcast_interval: 5 * SECOND,
                    },
                );
            }
            leader_idx = (leader_idx + 1) % self.servers.len();
        }

        // start
        self.app.gen_txn();
    }

    fn find_leader_by_id(servers: &mut Vec<Server>, region_id: u64) -> &mut Peer {
        servers
            .iter_mut()
            .find(|s| {
                s.peers
                    .get(&region_id)
                    .map(|p| p.role == Role::Leader)
                    .unwrap_or(false)
            })
            .unwrap()
            .peers
            .get_mut(&region_id)
            .unwrap()
    }

    fn find_followers_by_id(servers: &mut Vec<Server>, region_id: u64) -> Vec<&mut Peer> {
        servers
            .iter_mut()
            .filter_map(|s| {
                s.peers
                    .get_mut(&region_id)
                    .filter(|p| p.role == Role::Follower)
            })
            .collect()
    }

    fn find_server_by_id(servers: &mut Vec<Server>, server_id: u64) -> &mut Server {
        servers
            .iter_mut()
            .find(|s| s.server_id == server_id)
            .unwrap()
    }

    fn find_client_by_id(&mut self, client_id: u64) -> &mut Client {
        self.clients.iter_mut().find(|c| c.id == client_id).unwrap()
    }

    fn find_client_by_zone(&mut self, zone: Zone) -> &mut Client {
        self.clients.iter_mut().find(|c| c.zone == zone).unwrap()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Zone {
    AZ1,
    AZ2,
    AZ3,
}

impl Zone {
    fn rand_zone() -> Self {
        match rand::random::<u64>() % 3 {
            0 => Zone::AZ1,
            1 => Zone::AZ2,
            2 => Zone::AZ3,
            _ => panic!("invalid zone"),
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum Role {
    Leader,
    Follower,
}

#[derive(PartialEq, Eq)]
struct Peer {
    role: Role,
    server_id: u64,
    region_id: u64,
    // start_ts
    lock_cf: HashSet<u64>,
    // for leader, we assume the apply process are all fast enough.
    resolved_ts: u64,
    // for all roles
    safe_ts: u64,
    advance_interval: Time,
    broadcast_interval: Time,
}

impl Peer {
    fn update_resolved_ts(&mut self, events: Events) {
        assert!(self.role == Role::Leader);
        let min_lock = *self.lock_cf.iter().min().unwrap_or(&u64::MAX);
        let new_resolved_ts = min(now(), min_lock);
        assert!(new_resolved_ts >= self.resolved_ts);
        self.resolved_ts = new_resolved_ts;
        self.safe_ts = self.resolved_ts;

        let this_region_id = self.region_id;
        events.borrow_mut().push(Event::new(
            now() + self.advance_interval,
            EventType::ResolvedTsUpdate,
            Box::new(move |model: &mut Model| {
                let this = Model::find_leader_by_id(&mut model.servers, this_region_id);
                this.update_resolved_ts(model.events.clone());
            }),
        ));
    }

    fn broadcast_safe_ts(&mut self, events: Events) {
        assert!(self.role == Role::Leader);

        let this_region_id = self.region_id;
        let new_safe_ts = self.safe_ts;

        // broadcast
        let mut events = events.borrow_mut();
        events.push(Event::new(
            now() + rpc_latency(true),
            EventType::BroadcastSafeTs,
            Box::new(move |model: &mut Model| {
                for follower in Model::find_followers_by_id(&mut model.servers, this_region_id) {
                    assert!(follower.safe_ts <= new_safe_ts);
                    follower.safe_ts = new_safe_ts;
                }
            }),
        ));

        // schedule next
        events.push(Event::new(
            now() + self.broadcast_interval,
            EventType::BroadcastSafeTs,
            Box::new(move |model: &mut Model| {
                let this = Model::find_leader_by_id(&mut model.servers, this_region_id);
                this.broadcast_safe_ts(model.events.clone());
            }),
        ));
    }
}

struct Server {
    peers: HashMap<u64, Peer>,
    zone: Zone,
    server_id: u64,
    events: Events,
    // Vec<(accept time, task)>
    read_task_queue: VecDeque<(Time, Request)>,
    read_workers: Vec<Option<Request>>,
    read_schedule_wait_stat: Histogram<Time>,
    read_timeout: Time,
    // a read request will abort at this time if it cannot finish in time
    error_count: u64,

    write_task_queue: VecDeque<(Time, Request)>,
    write_workers: Vec<Option<Request>>,
    write_schedule_wait_stat: Histogram<Time>,
}

impl Server {
    fn new(
        zone: Zone,
        events: Events,
        num_read_workers: usize,
        num_write_workers: usize,
        read_timeout: Time,
    ) -> Self {
        let ret = Self {
            server_id: SERVER_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            peers: HashMap::new(),
            zone,
            events,
            read_task_queue: VecDeque::new(),
            read_workers: vec![None; num_read_workers],
            write_task_queue: VecDeque::new(),
            write_workers: vec![None; num_write_workers],
            read_schedule_wait_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
            read_timeout,
            error_count: 0,
            write_schedule_wait_stat: Histogram::<Time>::new_with_bounds(
                NANOSECOND,
                60 * SECOND,
                3,
            )
            .unwrap(),
        };
        ret
    }

    fn on_req(&mut self, task: Request) {
        match task.req_type {
            EventType::ReadRequest => {
                if self.read_workers.iter().all(|w| w.is_some()) {
                    // all busy
                    self.read_task_queue.push_back((now(), task));
                    return;
                }
                // some worker is idle, schedule now
                let worker_id = self.read_workers.iter().position(|w| w.is_none()).unwrap();
                self.handle_read(worker_id, task, now());
            }
            EventType::PrewriteRequest | EventType::CommitRequest => {
                // FCFS, no timeout
                if self.write_workers.iter().all(|w| w.is_some()) {
                    // all busy
                    self.write_task_queue.push_back((now(), task));
                    return;
                }
                // some worker is idle, schedule now
                let worker_id = self.write_workers.iter().position(|w| w.is_none()).unwrap();
                self.handle_write(worker_id, task, now());
            }
            _ => unreachable!(),
        }
    }

    // a worker is idle, schedule a task onto it now.
    // Invariant: the worker is idle.
    fn handle_read(&mut self, worker_id: usize, req: Request, accept_time: Time) {
        assert!(self.read_workers[worker_id].is_none());
        self.read_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();

        let peer = self.peers.get_mut(&req.region_id).unwrap();

        let task_size = req.size;
        let stale_read_ts = req.stale_read_ts;
        self.read_workers[worker_id] = Some(req);
        let this_server_id = self.server_id;

        // safe ts check
        if let Some(stale_read_ts) = stale_read_ts {
            if stale_read_ts > peer.safe_ts {
                // schedule next task in queue
                let task = self.read_workers[worker_id].take().unwrap();
                if let Some((accept_time, task)) = self.read_task_queue.pop_front() {
                    self.handle_read(worker_id, task, accept_time);
                }

                // return DataIsNotReady error
                self.events.borrow_mut().push(Event::new(
                    now() + rpc_latency(false),
                    EventType::Response,
                    Box::new(move |model: &mut Model| {
                        model
                            .find_client_by_id(task.client_id)
                            .on_resp(task, Some(Error::DataIsNotReady));
                    }),
                ));
                return;
            }
        }

        // timeout check
        if accept_time + self.read_timeout < now() + task_size {
            // will timeout. It tries for `read_timeout`, and then decide to abort.
            self.events.borrow_mut().push(Event::new(
                accept_time + self.read_timeout,
                EventType::ReadRequestTimeout,
                Box::new(move |model: &mut Model| {
                    let this = Model::find_server_by_id(&mut model.servers, this_server_id);
                    this.error_count += 1;

                    // schedule next task in queue
                    let task = this.read_workers[worker_id].take().unwrap();
                    if let Some((accept_time, task)) = this.read_task_queue.pop_front() {
                        this.handle_read(worker_id, task, accept_time);
                    }

                    // return response
                    model.events.borrow_mut().push(Event::new(
                        now() + rpc_latency(false),
                        EventType::Response,
                        Box::new(move |model: &mut Model| {
                            model
                                .find_client_by_id(task.client_id)
                                .on_resp(task, Some(Error::ReadTimeout));
                        }),
                    ));
                }),
            ));
            return;
        }

        // handle read
        self.events.borrow_mut().push(Event::new(
            now() + task_size,
            EventType::HandleRead,
            Box::new(move |model: &mut Model| {
                // schedule next task in queue
                let this: &mut Server =
                    Model::find_server_by_id(&mut model.servers, this_server_id);
                let task = this.read_workers[worker_id].take().unwrap();
                if let Some((accept_time, task)) = this.read_task_queue.pop_front() {
                    this.handle_read(worker_id, task, accept_time);
                }

                // return response
                model.events.borrow_mut().push(Event::new(
                    now() + rpc_latency(false),
                    EventType::Response,
                    Box::new(move |model: &mut Model| {
                        model.find_client_by_id(task.client_id).on_resp(task, None);
                    }),
                ));
            }),
        ));
    }

    fn handle_write(&mut self, worker_id: usize, req: Request, accept_time: Time) {
        assert!(self.write_workers[worker_id].is_none());
        let peer = self.peers.get_mut(&req.region_id).unwrap();
        assert!(peer.role == Role::Leader);
        self.write_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();
        let req_size = req.size;
        if req.req_type == EventType::PrewriteRequest {
            peer.lock_cf.insert(req.start_ts);
        }
        self.write_workers[worker_id] = Some(req);
        let this_server_id = self.server_id;
        self.events.borrow_mut().push(Event::new(
            now() + req_size,
            EventType::HandleWrite,
            Box::new(move |model: &mut Model| {
                let this: &mut Server =
                    Model::find_server_by_id(&mut model.servers, this_server_id);
                let task = this.write_workers[worker_id].take().unwrap();
                let peer = this.peers.get_mut(&task.region_id).unwrap();
                assert!(peer.role == Role::Leader);

                if task.req_type == EventType::CommitRequest {
                    assert!(peer.lock_cf.remove(&task.start_ts));
                }

                if let Some((accept_time, task)) = this.write_task_queue.pop_front() {
                    this.handle_write(worker_id, task, accept_time);
                }
                let this_zone = this.zone;

                let client = model.find_client_by_id(task.client_id);
                let remote = this_zone != client.zone;
                model.events.borrow_mut().push(Event::new(
                    now() + rpc_latency(remote),
                    EventType::Response,
                    Box::new(move |model: &mut Model| {
                        model.find_client_by_id(task.client_id).on_resp(task, None);
                    }),
                ));
            }),
        ));
    }
}

enum PeerSelectorState {
    StaleRead(StaleReaderState),
    NormalRead(NormalReaderState),
    Write(WriterState),
}

// local(stale) -> leader(normal) -> random follower(normal) -> error
enum StaleReaderState {
    LocalStale,
    LeaderNormal,
    RandomFollowerNormal,
}

enum NormalReaderState {
    Local,
    LeaderNormal,
    RandomFollowerNormal,
}

enum WriterState {
    Leader,
    LeaderFailed,
}

// TODO: this is a naive one, different from the one in client-go. Take care.
struct PeerSelector {
    state: PeerSelectorState,
    // already tried to perform non-stale read on these servers, no matter they are leader or follower
    server_ids_tried_for_normal_read: HashSet<u64>,
    local_zone: Zone,
}

impl PeerSelector {
    fn new(local: Zone, req: &Request) -> Self {
        let state = match req.req_type {
            EventType::ReadRequest => {
                if req.stale_read_ts.is_some() {
                    PeerSelectorState::StaleRead(StaleReaderState::LocalStale)
                } else {
                    PeerSelectorState::NormalRead(NormalReaderState::Local)
                }
            }
            EventType::PrewriteRequest | EventType::CommitRequest => {
                PeerSelectorState::Write(WriterState::Leader)
            }
            _ => unreachable!(),
        };
        Self {
            state: state,
            server_ids_tried_for_normal_read: HashSet::new(),
            local_zone: local,
        }
    }

    fn next<'a>(
        &mut self,
        servers: &'a mut Vec<Server>,
        req: &mut Request,
    ) -> Option<&'a mut Server> {
        match &mut self.state {
            PeerSelectorState::StaleRead(state) => match state {
                StaleReaderState::LocalStale => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    assert!(req.stale_read_ts.is_some());
                    *state = StaleReaderState::LeaderNormal;
                    let s = servers
                        .iter_mut()
                        .find(|s| s.zone == self.local_zone)
                        .unwrap();
                    Some(s)
                }
                StaleReaderState::LeaderNormal => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    req.stale_read_ts = None;
                    *state = StaleReaderState::RandomFollowerNormal;
                    let leader = Model::find_leader_by_id(servers, req.region_id);
                    let server_id = leader.server_id;
                    let s = Model::find_server_by_id(servers, server_id);
                    self.server_ids_tried_for_normal_read.insert(s.server_id);
                    Some(s)
                }
                StaleReaderState::RandomFollowerNormal => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    assert!(req.stale_read_ts.is_none());
                    let mut rng = rand::thread_rng();
                    let follower = Model::find_followers_by_id(servers, req.region_id)
                        .into_iter()
                        .filter(|s| !self.server_ids_tried_for_normal_read.contains(&s.server_id))
                        .choose(&mut rng);
                    if let Some(ref follower) = follower {
                        self.server_ids_tried_for_normal_read
                            .insert(follower.server_id);
                    }
                    let server_id = follower.map(|f| f.server_id);
                    server_id.map(|id| Model::find_server_by_id(servers, id))
                }
            },
            PeerSelectorState::Write(state) => match state {
                WriterState::Leader => {
                    let leader = Model::find_leader_by_id(servers, req.region_id);
                    let server_id = leader.server_id;
                    let s = Model::find_server_by_id(servers, server_id);
                    *state = WriterState::LeaderFailed;
                    Some(s)
                }
                WriterState::LeaderFailed => None,
            },
            PeerSelectorState::NormalRead(state) => match state {
                NormalReaderState::Local => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    let s = servers.iter_mut().find(|s| s.zone == self.local_zone);
                    if let Some(s) = &s {
                        self.server_ids_tried_for_normal_read.insert(s.server_id);
                    }
                    self.state = PeerSelectorState::NormalRead(NormalReaderState::LeaderNormal);
                    s
                }
                NormalReaderState::LeaderNormal => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    let leader = Model::find_leader_by_id(servers, req.region_id);
                    let server_id = leader.server_id;
                    let s = Model::find_server_by_id(servers, server_id);
                    self.server_ids_tried_for_normal_read.insert(s.server_id);
                    self.state =
                        PeerSelectorState::NormalRead(NormalReaderState::RandomFollowerNormal);
                    Some(s)
                }
                NormalReaderState::RandomFollowerNormal => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    let mut rng = rand::thread_rng();
                    let follower = Model::find_followers_by_id(servers, req.region_id)
                        .into_iter()
                        .filter(|s| !self.server_ids_tried_for_normal_read.contains(&s.server_id))
                        .choose(&mut rng);
                    if let Some(ref follower) = follower {
                        self.server_ids_tried_for_normal_read
                            .insert(follower.server_id);
                    }
                    let server_id = follower.map(|f| f.server_id);
                    server_id.map(|server_id| Model::find_server_by_id(servers, server_id))
                }
            },
        }
    }
}

struct Client {
    zone: Zone,
    id: u64,
    events: Events,
    // req_id -> (start_time, replica selector)
    pending_tasks: HashMap<u64, (Time, Rc<RefCell<PeerSelector>>)>,
    latency_stat: Histogram<Time>,
    error_latency_stat: HashMap<Error, Histogram<Time>>,
}

impl Client {
    fn new(zone: Zone, events: Events) -> Self {
        Self {
            id: CLIENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            zone,
            events,
            pending_tasks: HashMap::new(),
            latency_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap(),
            error_latency_stat: HashMap::new(),
        }
    }

    // app sends a req to client
    fn on_req(&mut self, mut req: Request) {
        req.client_id = self.id;
        let now = now();
        let selector = Rc::new(RefCell::new(PeerSelector::new(self.zone, &req)));
        self.pending_tasks
            .insert(req.req_id, (now, selector.clone()));
        self.issue_request(req, selector);
    }

    // send the req to the appropriate peer. If all peers have been tried, return error to app.
    fn issue_request(&mut self, mut req: Request, selector: Rc<RefCell<PeerSelector>>) {
        // we should decide the target *now*, but to access the server list in the model, we decide when
        // the event the rpc is to be accepted by the server.
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                let server = selector.borrow_mut().next(&mut model.servers, &mut req);
                if let Some(server) = server {
                    server.on_req(req);
                } else {
                    // no server available, return error
                    model.events.borrow_mut().push(Event::new(
                        now() + rpc_latency(false),
                        EventType::AppResp,
                        Box::new(move |model: &mut Model| {
                            model.app.on_resp(req, Some(Error::RegionUnavailable));
                        }),
                    ));
                }
            }),
        ));
    }

    fn on_resp(&mut self, req: Request, error: Option<Error>) {
        let (start_time, selector) = self.pending_tasks.get(&req.req_id).unwrap();
        self.latency_stat
            .record((now() - start_time) / NANOSECOND)
            .unwrap();

        if error.is_some() {
            self.error_latency_stat
                .entry(error.unwrap())
                .or_insert_with(|| {
                    Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap()
                })
                .record((now() - start_time) / NANOSECOND)
                .unwrap();
            // retry other peers
            self.issue_request(req, selector.clone());
        } else {
            // success. respond to app
            self.pending_tasks.remove(&req.req_id);
            self.events.borrow_mut().push(Event::new(
                now() + rpc_latency(false),
                EventType::AppResp,
                Box::new(move |model: &mut Model| {
                    model.app.on_resp(req, error);
                }),
            ));
        }
    }
}

struct App {
    events: Events,
    // the arrival rate of requests
    // in requests per second
    rng: StdRng,
    // the exponential distribution of the interval between two requests
    rate_exp_dist: Exp<f64>,
    // start_ts => transaction.
    pending_transactions: HashMap<Time, Transaction>,
    txn_duration_stat: Histogram<Time>,
    read_staleness: Option<Time>,
    region_num: u64,
    retry_count: u64,
}

enum CommitPhase {
    // For read-write transactions.
    NotYet,
    Prewriting,
    Committing,
    Committed,
    // read-only transaction doesn't need to commit.
    ReadOnly,
}

struct Transaction {
    zone: Zone,
    // start_ts, also the unique id of transactions
    start_ts: u64,
    // for read-only stale read transactions
    // invariant: must be smaller than start_ts and now().
    #[allow(unused)]
    stale_read_ts: Option<u64>,
    commit_ts: u64,
    remaining_queries: VecDeque<Request>,
    commit_phase: CommitPhase,
    prewrite_req: Option<Request>,
    commit_req: Option<Request>,
}

impl Transaction {
    fn new(
        num_queries: u64,
        read_only: bool,
        read_staleness: Option<Time>,
        region_num: u64,
    ) -> Self {
        assert!(read_staleness.is_none() || read_only);
        // we assume at least 1 query.
        assert!(num_queries > 0);
        let mut remaining_queries = VecDeque::new();
        let start_ts = now();
        let stale_read_ts = read_staleness.map(|staleness| now().saturating_sub(staleness));
        for _ in 0..num_queries {
            remaining_queries.push_back(Request::new(
                start_ts,
                stale_read_ts,
                EventType::ReadRequest,
                (rand::random::<u64>() % 5 + 1) * MILLISECOND,
                u64::MAX,
                rand::random::<u64>() % region_num,
            ));
        }
        let (mut prewrite_req, mut commit_req) = (None, None);
        if !read_only {
            let write_region = rand::random::<u64>() % region_num;
            prewrite_req = Some(Request::new(
                start_ts,
                None,
                EventType::PrewriteRequest,
                (rand::random::<u64>() % 5 + 1) * MILLISECOND,
                u64::MAX,
                write_region,
            ));
            commit_req = Some(Request::new(
                start_ts,
                None,
                EventType::CommitRequest,
                (rand::random::<u64>() % 5 + 1) * MILLISECOND,
                u64::MAX,
                write_region,
            ));
        }

        Self {
            zone: Zone::rand_zone(),
            start_ts,
            commit_ts: 0,
            remaining_queries,
            commit_phase: if read_only {
                CommitPhase::ReadOnly
            } else {
                CommitPhase::NotYet
            },
            stale_read_ts,
            prewrite_req,
            commit_req,
        }
    }
}

impl App {
    // req_rate: transactions per second
    fn new(events: Events, txn_rate: f64, read_staleness: Option<Time>, region_num: u64) -> Self {
        Self {
            events,
            rng: StdRng::from_seed(OsRng.gen()),
            rate_exp_dist: Exp::new(txn_rate).unwrap(),
            pending_transactions: HashMap::new(),
            txn_duration_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
            read_staleness,
            region_num,
            retry_count: 0,
        }
    }

    fn gen_txn(&mut self) {
        // 5% read-write, 95% read-only transactions.
        let read_only = rand::random::<u64>() % 20 > 0;
        let mut txn = Transaction::new(
            6,
            read_only,
            if read_only { self.read_staleness } else { None },
            self.region_num,
        );
        let zone = txn.zone;
        let req = txn.remaining_queries.pop_front().unwrap();
        self.pending_transactions.insert(txn.start_ts, txn);
        self.issue_request(zone, req);

        // Invariant: interval must be > 0, to avoid infinite loop.
        // And to make start_ts unique as start_ts is using now() directly
        let interval = ((self.rate_exp_dist.sample(&mut self.rng) * SECOND as f64) as Time).max(1);
        self.events.borrow_mut().push(Event::new(
            now() + interval,
            EventType::AppGen,
            Box::new(move |model: &mut Model| {
                model.app.gen_txn();
            }),
        ));
    }

    fn on_resp(&mut self, req: Request, error: Option<Error>) {
        if error.is_none() {
            let txn = self.pending_transactions.get_mut(&req.start_ts).unwrap();
            match txn.commit_phase {
                CommitPhase::ReadOnly => {
                    // no need to prewrite and commit. finish all read queries
                    txn.remaining_queries.pop_front();
                    if let Some(req) = txn.remaining_queries.pop_front() {
                        // send next query
                        let zone = txn.zone;
                        self.issue_request(zone, req);
                    } else {
                        txn.commit_phase = CommitPhase::Committed;
                        self.txn_duration_stat
                            .record((now() - txn.start_ts) / NANOSECOND)
                            .unwrap();
                        self.pending_transactions.remove(&req.start_ts);
                    }
                }
                CommitPhase::NotYet => {
                    if let Some(req) = txn.remaining_queries.pop_front() {
                        // send next query
                        let zone = txn.zone;
                        self.issue_request(zone, req);
                    } else {
                        txn.commit_phase = CommitPhase::Prewriting;
                        // send prewrite request
                        let zone = txn.zone;
                        let prewrite_req = txn.prewrite_req.take().unwrap();
                        self.issue_request(zone, prewrite_req);
                    }
                }
                CommitPhase::Prewriting => {
                    txn.commit_phase = CommitPhase::Committing;
                    // send commit request
                    let zone = txn.zone;
                    txn.commit_ts = now();
                    let commit_req = txn.commit_req.take().unwrap();
                    self.issue_request(zone, commit_req);
                }
                CommitPhase::Committing => {
                    txn.commit_phase = CommitPhase::Committed;
                    self.txn_duration_stat
                        .record((now() - txn.start_ts) / NANOSECOND)
                        .unwrap();
                    self.pending_transactions.remove(&req.start_ts);
                }
                CommitPhase::Committed => {
                    unreachable!();
                }
            }
        } else {
            // application retry immediately
            let txn = self.pending_transactions.get(&req.start_ts).unwrap();
            self.retry_count += 1;
            self.issue_request(txn.zone, req);
        }
    }

    fn issue_request(&mut self, zone: Zone, req: Request) {
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                model.find_client_by_zone(zone).on_req(req);
            }),
        ));
    }
}

fn now() -> Time {
    CURRENT_TIME.load(atomic::Ordering::SeqCst)
}

fn rpc_latency(remote: bool) -> u64 {
    let mut rng = rand::thread_rng();
    let t = if remote {
        REMOTE_NET_LATENCY_DIST.sample(&mut rng) as u64
    } else {
        LOCAL_NET_LATENCY_DIST.sample(&mut rng) as u64
    };
    if t > 60 * SECOND {
        panic!("invalid rpc latency: {}", t);
    }
    t
}

// a request, its content are permanent and immutable, even if it's sent to multiple servers.
#[derive(Clone)]
struct Request {
    start_ts: u64,
    stale_read_ts: Option<u64>,
    req_type: EventType,
    req_id: u64,
    client_id: u64,
    // the time needed to finish the task, in microsecond
    size: Time,
    region_id: u64,
}

impl Request {
    fn new(
        start_ts: u64,
        stale_read_ts: Option<u64>,
        req_type: EventType,
        size: Time,
        client_id: u64,
        region: u64,
    ) -> Self {
        Self {
            start_ts,
            stale_read_ts,
            req_type,
            req_id: TASK_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            client_id,
            size,
            region_id: region,
        }
    }
}

// a heap of events, sorted by the time of trigger
struct EventHeap {
    events: BinaryHeap<Event>,
}

impl EventHeap {
    fn new() -> Self {
        EventHeap {
            events: BinaryHeap::new(),
        }
    }

    fn peek(&self) -> Option<&Event> {
        self.events.peek()
    }

    fn pop(&mut self) -> Option<Event> {
        self.events.pop()
    }

    fn push(&mut self, event: Event) {
        assert!(now() <= event.trigger_time);
        self.events.push(event);
    }
}

#[allow(unused)]
struct Event {
    id: u64,
    trigger_time: Time,
    event_type: EventType,
    f: Box<dyn FnOnce(&mut Model)>,
}

impl Event {
    fn new(trigger_time: Time, event_type: EventType, f: Box<dyn FnOnce(&mut Model)>) -> Self {
        Event {
            id: EVENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            trigger_time,
            event_type,
            f,
        }
    }
}

impl Eq for Event {}

impl PartialEq<Self> for Event {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd<Self> for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.trigger_time
            .cmp(&other.trigger_time)
            .reverse()
            .then(self.id.cmp(&other.id).reverse())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum EventType {
    HandleRead,
    HandleWrite,
    ResolvedTsUpdate,
    BroadcastSafeTs,
    ReadRequest,
    ReadRequestTimeout,
    PrewriteRequest,
    CommitRequest,
    // server to client
    Response,
    // client to app
    AppResp,
    AppGen,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
enum Error {
    // server is busy - deadline exceeded
    ReadTimeout,
    // all servers are unavailable
    RegionUnavailable,
    DataIsNotReady,
}
