use hdrhistogram::Histogram;
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::cmp::Ordering;
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
        events: events.clone(),
        servers: vec![
            Server::new(Role::Leader, Zone::AZ1, events.clone(), 4, 2, 1 * SECOND),
            Server::new(Role::Follower, Zone::AZ2, events.clone(), 4, 2, 1 * SECOND),
            Server::new(Role::Follower, Zone::AZ3, events.clone(), 4, 2, 1 * SECOND),
        ],
        clients: vec![
            Client::new(Zone::AZ1, events.clone()),
            Client::new(Zone::AZ2, events.clone()),
            Client::new(Zone::AZ3, events.clone()),
        ],
        app: App::new(events.clone(), 1000.0),
    };

    model.app.gen_txn();
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
            "client {:?} mean: {}, p99: {}, {} errors",
            client.id,
            (client.latency_stat.mean() as u64).pretty_print(),
            (client.latency_stat.value_at_quantile(0.99) as u64).pretty_print(),
            client.error_latency_stat.len(),
        );
    }

    for server in model.servers {
        println!("server {}", server.server_id);
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
    app: App,
}

impl Model {
    fn find_server_by_id(&mut self, server_id: u64) -> &mut Server {
        self.servers
            .iter_mut()
            .find(|s| s.server_id == server_id)
            .unwrap()
    }

    fn find_client_by_id(&mut self, client_id: u64) -> &mut Client {
        self.clients.iter_mut().find(|c| c.id == client_id).unwrap()
    }

    fn find_server_by_zone(&mut self, zone: Zone) -> &mut Server {
        self.servers.iter_mut().find(|s| s.zone == zone).unwrap()
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

#[derive(PartialEq)]
enum Role {
    Leader,
    Follower,
}

struct Server {
    role: Role,
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
        role: Role,
        zone: Zone,
        events: Events,
        num_read_workers: usize,
        num_write_workers: usize,
        read_timeout: Time,
    ) -> Self {
        Self {
            server_id: SERVER_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            role,
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
        }
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
                self.read_schedule_wait_stat.record(0).unwrap();
                let worker_id = self.read_workers.iter().position(|w| w.is_none()).unwrap();
                self.schedule_to_read_worker(worker_id, task, now());
            }
            EventType::PrewriteRequest | EventType::CommitRequest => {
                // FCFS, no timeout
                assert!(self.role == Role::Leader);
                if self.write_workers.iter().all(|w| w.is_some()) {
                    // all busy
                    self.write_task_queue.push_back((now(), task));
                    return;
                }
                // some worker is idle, schedule now
                self.write_schedule_wait_stat.record(0).unwrap();
                let worker_id = self.write_workers.iter().position(|w| w.is_none()).unwrap();
                self.schedule_to_write_worker(worker_id, task, now());
            }
            _ => unreachable!(),
        }
    }

    // a worker is idle, schedule a task onto it now.
    // Invariant: the worker is idle.
    fn schedule_to_read_worker(&mut self, worker_id: usize, req: Request, accept_time: Time) {
        assert!(self.read_workers[worker_id].is_none());
        self.read_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();

        let task_size = req.size;
        self.read_workers[worker_id] = Some(req);
        let this_server_id = self.server_id;
        if accept_time + self.read_timeout < now() + task_size {
            // will timeout. It tries for `read_timeout`, and then decide to abort.
            self.events.borrow_mut().push(Event::new(
                accept_time + self.read_timeout,
                EventType::ReadRequestTimeout,
                Box::new(move |model: &mut Model| {
                    let this = model.find_server_by_id(this_server_id);
                    this.error_count += 1;
                    let task = this.read_workers[worker_id].take().unwrap();
                    if let Some((accept_time, task)) = this.read_task_queue.pop_front() {
                        this.schedule_to_read_worker(worker_id, task, accept_time);
                    }
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

        self.events.borrow_mut().push(Event::new(
            now() + task_size,
            EventType::HandleRead,
            Box::new(move |model: &mut Model| {
                let this: &mut Server = model.find_server_by_id(this_server_id);
                let task = this.read_workers[worker_id].take().unwrap();
                if let Some((accept_time, task)) = this.read_task_queue.pop_front() {
                    this.schedule_to_read_worker(worker_id, task, accept_time);
                }
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

    fn schedule_to_write_worker(&mut self, worker_id: usize, req: Request, accept_time: Time) {
        assert!(self.write_workers[worker_id].is_none());
        self.write_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();
        let req_size = req.size;
        self.write_workers[worker_id] = Some(req);
        let this_server_id = self.server_id;
        self.events.borrow_mut().push(Event::new(
            now() + req_size,
            EventType::Response,
            Box::new(move |model: &mut Model| {
                let this: &mut Server = model.find_server_by_id(this_server_id);
                let task = this.write_workers[worker_id].take().unwrap();
                if let Some((accept_time, task)) = this.write_task_queue.pop_front() {
                    this.schedule_to_write_worker(worker_id, task, accept_time);
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
    Read(ReaderState),
    Write(WriterState),
}

// local -> leader -> random follower -> error
enum ReaderState {
    Local,
    Leader,
    RandomFollower,
}

enum WriterState {
    Leader,
    LeaderFailed,
}

// TODO: this is a naive one, different from the one in client-go. Take care.
struct PeerSelector {
    state: PeerSelectorState,
    follower_ids_tried: HashSet<u64>,
    local_zone: Zone,
}

impl PeerSelector {
    fn new(local: Zone, req: &Request) -> Self {
        let state = match req.req_type {
            EventType::ReadRequest => PeerSelectorState::Read(ReaderState::Local),
            EventType::PrewriteRequest | EventType::CommitRequest => {
                PeerSelectorState::Write(WriterState::Leader)
            }
            _ => unreachable!(),
        };
        Self {
            state: state,
            follower_ids_tried: HashSet::new(),
            local_zone: local,
        }
    }

    fn next<'a>(&mut self, servers: &'a mut Vec<Server>) -> Option<&'a mut Server> {
        match &mut self.state {
            PeerSelectorState::Read(state) => match state {
                ReaderState::Local => {
                    *state = ReaderState::Leader;
                    let s = servers
                        .iter_mut()
                        .find(|s| s.zone == self.local_zone)
                        .unwrap();
                    self.follower_ids_tried.insert(s.server_id);
                    Some(s)
                }
                ReaderState::Leader => {
                    *state = ReaderState::RandomFollower;
                    let s = servers.iter_mut().find(|s| s.role == Role::Leader).unwrap();
                    Some(s)
                }
                ReaderState::RandomFollower => {
                    let mut rng = rand::thread_rng();
                    let follower = servers
                        .iter_mut()
                        .filter(|s| s.role == Role::Follower)
                        .filter(|s| !self.follower_ids_tried.contains(&s.server_id))
                        .choose(&mut rng);
                    if let Some(ref follower) = follower {
                        self.follower_ids_tried.insert(follower.server_id);
                    }
                    follower
                }
            },
            PeerSelectorState::Write(state) => match state {
                WriterState::Leader => {
                    let s = servers.iter_mut().find(|s| s.role == Role::Leader).unwrap();
                    *state = WriterState::LeaderFailed;
                    Some(s)
                }
                WriterState::LeaderFailed => None,
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
    error_latency_stat: Histogram<Time>,
}

impl Client {
    fn new(zone: Zone, events: Events) -> Self {
        Self {
            id: CLIENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            zone,
            events,
            pending_tasks: HashMap::new(),
            latency_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap(),
            error_latency_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
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
    fn issue_request(&mut self, req: Request, selector: Rc<RefCell<PeerSelector>>) {
        // we should decide the target *now*, but to access the server list in the model, we decide when
        // the event the rpc is to be accepted by the server.
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                let server = selector.borrow_mut().next(&mut model.servers);
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
}

enum CommitPhase {
    NotYet,
    Prewriting,
    Committing,
    Committed,
}

struct Transaction {
    zone: Zone,
    start_ts: u64,
    commit_ts: u64,
    num_queries: u64,
    finished_queries: u64,
    commit_phase: CommitPhase,
}

impl Transaction {
    fn new(num_queries: u64) -> Self {
        Self {
            zone: Zone::rand_zone(),
            start_ts: now(),
            commit_ts: 0,
            num_queries,
            finished_queries: 0,
            commit_phase: CommitPhase::NotYet,
        }
    }
}

impl App {
    fn new(events: Events, req_rate: f64) -> Self {
        Self {
            events,
            rng: StdRng::from_seed(OsRng.gen()),
            rate_exp_dist: Exp::new(req_rate).unwrap(),
            pending_transactions: HashMap::new(),
            txn_duration_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
        }
    }

    fn gen_txn(&mut self) {
        let txn = Transaction::new(3);
        let start_ts = txn.start_ts;
        let zone = txn.zone;
        self.pending_transactions.insert(txn.start_ts, txn);
        self.issue_new_read_request(start_ts, zone);

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
                CommitPhase::NotYet => {
                    assert!(txn.finished_queries < txn.num_queries);
                    txn.finished_queries += 1;
                    if txn.finished_queries == txn.num_queries {
                        txn.commit_phase = CommitPhase::Prewriting;
                        // send prewrite request
                        let zone = txn.zone;
                        let prewrite_req = Request::new(
                            txn.start_ts,
                            EventType::PrewriteRequest,
                            (rand::random::<u64>() % 5 + 1) * MILLISECOND,
                            u64::MAX,
                        );
                        self.issue_request(zone, prewrite_req);
                    } else {
                        // send next query
                        let start_ts = txn.start_ts;
                        let zone = txn.zone;
                        self.issue_new_read_request(start_ts, zone);
                    }
                }
                CommitPhase::Prewriting => {
                    txn.commit_phase = CommitPhase::Committing;
                    // send commit request
                    let zone = txn.zone;
                    txn.commit_ts = now();
                    let commit_req = Request::new(
                        txn.start_ts,
                        EventType::CommitRequest,
                        (rand::random::<u64>() % 5 + 1) * MILLISECOND,
                        u64::MAX,
                    );
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

    fn issue_new_read_request(&mut self, start_ts: u64, zone: Zone) {
        // size of 1 - 5 ms
        let req = Request::new(
            start_ts,
            EventType::ReadRequest,
            (rand::random::<u64>() % 5 + 1) * MILLISECOND,
            u64::MAX,
        );
        let zone = zone;
        self.issue_request(zone, req);
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
    req_type: EventType,
    req_id: u64,
    client_id: u64,
    // the time needed to finish the task, in microsecond
    size: Time,
}

impl Request {
    fn new(start_ts: u64, req_type: EventType, size: Time, client_id: u64) -> Self {
        Self {
            start_ts,
            req_type,
            req_id: TASK_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            client_id,
            size,
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

#[derive(Debug, Copy, Clone)]
enum EventType {
    HandleRead,
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

#[derive(Debug, Copy, Clone)]
enum Error {
    // server is busy - deadline exceeded
    ReadTimeout,
    // all servers are unavailable
    RegionUnavailable,
}
