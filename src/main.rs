use hdrhistogram::Histogram;
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;

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
const MICROSECOND: Time = 1_000;
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
    // 跨AZ
    static ref REMOTE_NET_LATENCY_DIST: LogNormal<f64> = {
        let mean = 0.5 * MILLISECOND as f64;
        let std_dev = 0.1 * MILLISECOND as f64;
        let location = (mean.powi(2) / (mean.powi(2) + std_dev.powi(2)).sqrt()).ln();
        let scale = (1.0 + std_dev.powi(2) / mean.powi(2)).ln().sqrt();
        LogNormal::new(location, scale).unwrap()
    };

    // AZ内
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
            Server::new(Role::Leader, Zone::AZ1, events.clone(), 4),
            Server::new(Role::Follower, Zone::AZ2, events.clone(), 4),
            Server::new(Role::Follower, Zone::AZ3, events.clone(), 4),
        ],
        clients: vec![
            Client::new(Zone::AZ1, events.clone()),
            Client::new(Zone::AZ2, events.clone()),
            Client::new(Zone::AZ3, events.clone()),
        ],
        app: App::new(events.clone(), 3_000.0),
    };

    model.app.gen();
    let max_time = 60 * SECOND;

    loop {
        // get current event time
        if events
            .borrow()
            .peek()
            .map(|e| e.trigger_time > max_time)
            .unwrap_or(true)
        {
            break;
        }
        let mut events_mut = events.borrow_mut();
        let event = events_mut.pop().unwrap();
        CURRENT_TIME.store(event.trigger_time, atomic::Ordering::SeqCst);
        drop(events_mut);
        (event.f)(&mut model);
    }

    println!(
        "simulation time: {}, generated {} tasks, finished {} tasks",
        now().pretty_print(),
        TASK_COUNTER.load(atomic::Ordering::SeqCst),
        model
            .clients
            .iter()
            .map(|c| c.latency_stat.len())
            .sum::<u64>()
    );

    for client in model.clients {
        println!(
            "client {:?} mean: {}, p99: {}",
            client.id,
            (client.latency_stat.mean() as u64).pretty_print(),
            (client.latency_stat.value_at_quantile(0.99) as u64).pretty_print()
        );
    }

    for server in model.servers {
        println!(
            "server {:?} schedule wait mean: {}, p99: {}",
            server.server_id,
            (server.schedule_wait_stat.mean() as u64).pretty_print(),
            (server.schedule_wait_stat.value_at_quantile(0.99) as u64).pretty_print(),
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
    fn from_id(id: u64) -> Zone {
        match id {
            0 => Zone::AZ1,
            1 => Zone::AZ2,
            2 => Zone::AZ3,
            _ => panic!("invalid zone"),
        }
    }
}

enum Role {
    Leader,
    Follower,
}

struct Server {
    role: Role,
    zone: Zone,
    server_id: u64,
    events: Events,
    // Vec<(time when enqueue, task)>
    task_queue: VecDeque<(Time, Request)>,
    workers: Vec<Option<Request>>,
    schedule_wait_stat: Histogram<Time>,
}

impl Server {
    fn new(role: Role, zone: Zone, events: Events, num_workers: usize) -> Self {
        Self {
            server_id: SERVER_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            role,
            zone,
            events,
            task_queue: VecDeque::new(),
            workers: vec![None; num_workers],
            schedule_wait_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
        }
    }

    fn on_req(&mut self, task: Request) {
        if self.workers.iter().all(|w| w.is_some()) {
            // busy
            self.task_queue.push_back((now(), task));
            return;
        }
        self.schedule_wait_stat.record(0).unwrap();
        // some work is idle, schedule now
        let worker_id = self.workers.iter().position(|w| w.is_none()).unwrap();
        let task_size = task.size;
        self.workers[worker_id] = Some(task);

        // prepare for wakeup event
        let current_time = now();
        let this_server_id = self.server_id;
        self.events.borrow_mut().push(Event::new(
            current_time + task_size,
            EventType::TaskFinish,
            Box::new(move |model: &mut Model| {
                let this: &mut Server = model.find_server_by_id(this_server_id);
                let task = this.workers[worker_id].take().unwrap();
                if let Some((enqueue_time, task)) = this.task_queue.pop_front() {
                    // invariant: there must be an idle worker, must schedule now
                    this.schedule_wait_stat
                        .record(now() - enqueue_time)
                        .unwrap();
                    this.on_req(task);
                }
                model.events.borrow_mut().push(Event::new(
                    now() + rpc_latency(false),
                    EventType::Response,
                    Box::new(move |model: &mut Model| {
                        model.find_client_by_id(task.client_id).on_resp(task);
                    }),
                ));
            }),
        ));
    }
}

struct Client {
    zone: Zone,
    id: u64,
    events: Events,
    // task_id -> start_time
    pending_tasks: HashMap<u64, Time>,
    latency_stat: Histogram<Time>,
}

impl Client {
    fn new(zone: Zone, events: Events) -> Self {
        Self {
            id: CLIENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            zone,
            events,
            pending_tasks: HashMap::new(),
            latency_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap(),
        }
    }

    // app sends a req to client
    fn on_req(&mut self, mut req: Request) {
        req.client_id = self.id;
        // sends to local server
        let now = now();
        self.pending_tasks.insert(req.req_id, now);
        let zone = self.zone;
        self.events.borrow_mut().push(Event::new(
            now + rpc_latency(false),
            EventType::Request,
            Box::new(move |model: &mut Model| {
                model.find_server_by_zone(zone).on_req(req);
            }),
        ));
    }

    fn on_resp(&mut self, resp: Request) {
        let start_time = self.pending_tasks.remove(&resp.req_id).unwrap();
        self.latency_stat
            .record((now() - start_time) / NANOSECOND)
            .unwrap();
    }
}

struct App {
    events: Events,
    // the arrival rate of requests
    // in requests per second
    rng: StdRng,
    // the exponential distribution of the interval between two requests
    rate_exp_dist: Exp<f64>,
}

impl App {
    fn new(events: Events, req_rate: f64) -> Self {
        Self {
            events,
            rng: StdRng::from_seed(OsRng.gen()),
            rate_exp_dist: Exp::new(req_rate).unwrap(),
        }
    }

    fn gen(&mut self) {
        // size of 1 - 5 ms
        let req = Request::new((rand::random::<u64>() % 5 + 1) * MILLISECOND, u64::MAX);
        let mut events = self.events.borrow_mut();
        let zone = Zone::from_id(rand::random::<u64>() % 3);
        events.push(Event::new(
            now() + rpc_latency(false),
            EventType::AppReq,
            Box::new(move |model: &mut Model| {
                model.find_client_by_zone(zone).on_req(req);
            }),
        ));

        // interval must be > 0, to avoid infinite loop
        let interval = ((self.rate_exp_dist.sample(&mut self.rng) * SECOND as f64) as Time).max(1);
        events.push(Event::new(
            now() + interval,
            EventType::AppGen,
            Box::new(move |model: &mut Model| {
                model.app.gen();
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

#[derive(Clone)]
struct Request {
    req_id: u64,
    client_id: u64,
    size: Time, // the time needed to finish the task, in microsecond
}

impl Request {
    fn new(size: Time, client_id: u64) -> Self {
        Self {
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

#[derive(Debug)]
enum EventType {
    TaskFinish,
    Request,
    Response,
    AppReq,
    AppGen,
}
