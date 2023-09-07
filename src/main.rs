use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;

use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp};

type Events = Rc<RefCell<EventHeap>>;

type Time = u64;

const NANOSECOND: Time = 1;
const MICROSECOND: Time = 1_000;
const MILLISECOND: Time = 1_000 * MICROSECOND;
const SECOND: Time = 1_000 * MILLISECOND;

static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);
static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);
static SERVER_COUNTER: AtomicU64 = AtomicU64::new(0);
static CLIENT_COUNTER: AtomicU64 = AtomicU64::new(0);
static CURRENT_TIME: AtomicU64 = AtomicU64::new(0);

fn main() {
    let events: Rc<RefCell<EventHeap>> = Rc::new(RefCell::new(EventHeap::new()));

    let mut model = Model {
        events: events.clone(),
        servers: [
            Server::new(Role::Leader, Zone::AZ1, events.clone(), 4),
            Server::new(Role::Follower, Zone::AZ2, events.clone(), 4),
            Server::new(Role::Follower, Zone::AZ3, events.clone(), 4),
        ],
        clients: [
            Client::new(Zone::AZ1, events.clone()),
            Client::new(Zone::AZ2, events.clone()),
            Client::new(Zone::AZ3, events.clone()),
        ],
        app: App::new(events.clone(), 3_000.0),
    };

    model.app.gen();
    let max_time = 1 * SECOND;

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
}

struct Model {
    events: Events,
    servers: [Server; 3],
    clients: [Client; 3],
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
    task_queue: VecDeque<Request>,
    workers: Vec<Option<Request>>,
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
        }
    }

    fn schedule_task(&mut self, task: Request) {
        if self.workers.iter().all(|w| w.is_some()) {
            // busy
            self.task_queue.push_back(task);
            return;
        }

        // some worker is idle
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
                if let Some(task) = this.task_queue.pop_front() {
                    this.schedule_task(task);
                }
                model.events.borrow_mut().push(Event::new(
                    now() + rpc_latency(),
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
}

impl Client {
    fn new(zone: Zone, events: Events) -> Self {
        Self {
            id: CLIENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            zone,
            events,
            pending_tasks: HashMap::new(),
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
            now + rpc_latency(),
            EventType::Request,
            Box::new(move |model: &mut Model| {
                model.find_server_by_zone(zone).schedule_task(req);
            }),
        ));
    }

    fn on_resp(&mut self, resp: Request) {
        let _start_time = self.pending_tasks.remove(&resp.req_id).unwrap();
        println!(
            "resp {} finished, takes {} ms, {} in flight",
            resp.req_id,
            (now() - _start_time) / (MILLISECOND),
            self.pending_tasks.len()
        );
    }
}

struct App {
    events: Events,
    // the arrival rate of requests
    // in requests per second
    req_rate: f64,
    rng: StdRng,
    // the exponential distribution of the interval between two requests
    rate_exp_dist: Exp<f64>,
}

impl App {
    fn new(events: Events, req_rate: f64) -> Self {
        Self {
            events,
            req_rate,
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
            now() + rpc_latency(),
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

fn rpc_latency() -> Time {
    // 0~200 us
    rand::random::<u64>() % 200 * MICROSECOND
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
    created_time: Time,
    trigger_time: Time,
    event_type: EventType,
    f: Box<dyn FnOnce(&mut Model)>,
}

impl Event {
    fn new(trigger_time: Time, event_type: EventType, f: Box<dyn FnOnce(&mut Model)>) -> Self {
        Event {
            id: EVENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            created_time: now(),
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
