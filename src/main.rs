use hdrhistogram::Histogram;
use indicatif::ProgressBar;
use lazy_static::lazy_static;
use plotters::backend::SVGBackend;
use plotters::chart::SeriesLabelPosition;
use plotters::coord::Shift;
use plotters::drawing::DrawingArea;
use plotters::element::PathElement;
use plotters::prelude::{ChartBuilder, Color, FontDesc, LineSeries, RGBColor, WHITE};
use std::cell::RefCell;
use std::cmp::{max, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::default::Default;
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::rc::Rc;
use std::sync::atomic;
use std::sync::atomic::AtomicU64;

use rand::prelude::IteratorRandom;
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal};
use tikv_sim::TimeTrait;
use tikv_sim::{Time, *};

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

lazy_static! {
    static ref M: MetricsRecorder = MetricsRecorder::new();
}

type Events = Rc<RefCell<EventHeap>>;

static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);
static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);
static SERVER_COUNTER: AtomicU64 = AtomicU64::new(0);
static CLIENT_COUNTER: AtomicU64 = AtomicU64::new(0);

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
struct Config {
    num_replica: usize,
    num_region: u64,
    num_servers: u64,
    max_time: Time,
    metrics_interval: Time,
    enable_trace: bool,

    server_config: ServerConfig,
    client_config: ClientConfig,
    app_config: AppConfig,
}

#[derive(Clone)]
struct ServerConfig {
    enable_async_commit: bool,
    num_read_workers: usize,
    num_write_workers: usize,
    read_timeout: Time,
    advance_interval: Time,
    broadcast_interval: Time,
}

#[derive(Clone)]
struct ClientConfig {
    max_execution_time: Time,
}

#[derive(Clone)]
struct AppConfig {
    retry: bool,
    // transactions per second
    txn_rate: f64,
    read_staleness: Option<Time>,
    read_size_fn: Rc<dyn Fn() -> Time>,
    prewrite_size_fn: Rc<dyn Fn() -> Time>,
    commit_size_fn: Rc<dyn Fn() -> Time>,
    num_queries_fn: Rc<dyn Fn() -> u64>,
    read_only_ratio: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let events: Rc<RefCell<EventHeap>> = Rc::new(RefCell::new(EventHeap::new()));
    let config = Config {
        num_replica: 3,
        num_region: 10000,
        num_servers: 12,
        max_time: 500 * SECOND,
        metrics_interval: SECOND,
        enable_trace: false,
        server_config: ServerConfig {
            enable_async_commit: true,
            num_read_workers: 1,
            num_write_workers: 1,
            read_timeout: SECOND,
            advance_interval: 5 * SECOND,
            broadcast_interval: 5 * SECOND,
        },
        client_config: ClientConfig {
            max_execution_time: 10 * SECOND,
        },
        app_config: AppConfig {
            retry: false,
            txn_rate: 1200.0,
            read_staleness: Some(12 * SECOND),
            read_size_fn: Rc::new(|| (rand::random::<u64>() % 5 + 1) * MILLISECOND),
            prewrite_size_fn: Rc::new(|| (rand::random::<u64>() % 30 + 1) * MILLISECOND),
            commit_size_fn: Rc::new(|| (rand::random::<u64>() % 20 + 1) * MILLISECOND),
            num_queries_fn: Rc::new(|| 5),
            read_only_ratio: 0.99,
        },
    };
    assert_eq!(config.metrics_interval, SECOND, "why bother");
    assert!(config.num_servers >= 3); // at least 1 server per AZ.
    assert_eq!(
        config.num_replica, 3,
        "not supported yet, consider its match with AZs"
    );

    let mut model = Model {
        events: events.clone(),
        num_replica: config.num_replica,
        metrics_interval: config.metrics_interval,
        servers: (0..config.num_servers)
            .map(|id| Server::new(Zone::from_id(id), events.clone(), &config))
            .collect(),
        clients: vec![
            Client::new(Zone::AZ1, events.clone(), &config),
            Client::new(Zone::AZ2, events.clone(), &config),
            Client::new(Zone::AZ3, events.clone(), &config),
        ],
        app: App::new(events.clone(), &config),
    };
    model.init(&config);
    model.inject_high_pressure();

    let bar = ProgressBar::new(config.max_time / SECOND);
    loop {
        let mut events_mut = events.borrow_mut();
        let event = match events_mut.pop() {
            None => break,
            Some(e) => {
                if e.trigger_time > config.max_time {
                    break;
                }
                e
            }
        };
        // time cannot go back
        assert!(now() <= event.trigger_time);
        bar.set_position(now() / SECOND);
        CURRENT_TIME.store(event.trigger_time, atomic::Ordering::SeqCst);
        drop(events_mut);
        (event.f)(&mut model);
    }
    bar.finish();
    draw_metrics(&config)?;
    Ok(())
}

fn draw_metrics(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let num_graphs = 20;
    let root = SVGBackend::new("0.svg", (1200, num_graphs * 300)).into_drawing_area();
    let children_area = root.split_evenly(((num_graphs as usize + 1) / 2, 2));
    let font = ("Jetbrains Mono", 15).into_font();
    let colors = [
        RGBColor(0x1f, 0x77, 0xb4),
        RGBColor(0xff, 0x7f, 0x0e),
        RGBColor(0x2c, 0xa0, 0x2c),
        RGBColor(0xd6, 0x27, 0x28),
        RGBColor(0x94, 0x67, 0xbd),
        RGBColor(0x8c, 0x56, 0x4b),
        RGBColor(0xe3, 0x77, 0xc2),
        RGBColor(0x7f, 0x7f, 0x7f),
        RGBColor(0xbc, 0xbd, 0x22),
        RGBColor(0x17, 0xbe, 0xcf),
        RGBColor(0xa0, 0x52, 0x2d),
        RGBColor(0x6a, 0x5a, 0xcd),
        RGBColor(0x20, 0xb2, 0xaa),
        RGBColor(0x00, 0x64, 0x00),
        RGBColor(0x8b, 0x00, 0x8b),
        RGBColor(0x5f, 0x9e, 0xa0),
        RGBColor(0x9a, 0xcd, 0x32),
        RGBColor(0xff, 0x45, 0x00),
        RGBColor(0xda, 0x70, 0xd6),
        RGBColor(0x00, 0xce, 0xd1),
        RGBColor(0x40, 0xe0, 0xd0),
        RGBColor(0xff, 0xd7, 0x00),
        RGBColor(0xad, 0xff, 0x2f),
        RGBColor(0xff, 0x69, 0xb4),
        RGBColor(0xcd, 0x5c, 0x5c),
        RGBColor(0x4b, 0x00, 0x82),
        RGBColor(0x87, 0xce, 0xeb),
        RGBColor(0x32, 0xcd, 0x32),
        RGBColor(0x6b, 0x8e, 0x23),
        RGBColor(0xff, 0xa5, 0x00),
    ];
    let hist_latency_opts = Opts::Hist(HistOpts::Percentiles {
        mean: true,
        p99: true,
        p9999: false,
    });
    let hist_count_opts = Opts::Hist(HistOpts::Count);
    let counter_opts = Opts::Counter;
    let gauge_opts = Opts::Gauge;
    let (ms_unit, ms_label) = (MILLISECOND, "ms");
    let mut chart_id = 0;

    let app_gen_txn_counter = M.get_metric_group("app_gen_txn_counter").unwrap();
    plot_chart(
        &children_area,
        &font,
        colors,
        counter_opts,
        &mut chart_id,
        "txn generated",
        app_gen_txn_counter,
        1,
        "",
    )?;

    let app_ok_txn_duration = M
        .get_metric_group("app_txn_duration")
        .unwrap()
        .filter_by_label(0, "ok");
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_count_opts,
        &mut chart_id,
        "OK transaction rate",
        app_ok_txn_duration,
        1,
        "",
    )?;

    let app_txn_latency = M
        .get_metric_group("app_txn_duration")
        .expect("no app txn duration records")
        .filter_by_label(0, "ok");
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "txn latency",
        app_txn_latency,
        ms_unit,
        ms_label,
    )?;

    let app_error_rate = M
        .get_metric_group("app_txn_duration")
        .unwrap()
        .filter_by_label(0, "fail");
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_count_opts,
        &mut chart_id,
        "app error rate",
        app_error_rate,
        1,
        "",
    )?;

    let kv_ok_latency = M
        .get_metric_group("kv_latency")
        .unwrap()
        .filter_by_label(0, "ok");
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_count_opts,
        &mut chart_id,
        "OK KV rate",
        kv_ok_latency.clone(),
        1,
        "",
    )?;

    let kv_fail_latency = M
        .get_metric_group("kv_latency")
        .unwrap()
        .filter_by_label(0, "fail");
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_count_opts,
        &mut chart_id,
        "KV error rate",
        kv_fail_latency.clone(),
        1,
        "",
    )?;

    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "KV OK latency",
        kv_ok_latency,
        ms_unit,
        ms_label,
    )?;

    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "KV error latency",
        kv_fail_latency,
        ms_unit,
        ms_label,
    )?;

    // If there are too many servers, we plot the aggregated metrics, otherwise plot per-server metrics.
    let server_read_queue_length;
    let server_write_queue_length;
    let server_max_resolved_ts_gap;
    let advance_resolved_ts_fail_count;
    let server_reader_utilization;
    let server_writer_utilization;
    let server_read_req_count;
    let server_write_req_count;
    let server_error_count;
    let server_read_schedule_wait;
    let server_write_schedule_wait;
    let server_rest_time_to_handle_read;
    if colors.len() >= cfg.num_servers as usize {
        server_read_queue_length = M.get_metric_group("server_read_queue_length").unwrap();
        server_write_queue_length = M.get_metric_group("server_write_queue_length").unwrap();
        server_max_resolved_ts_gap = M
            .get_metric_group("server_max_resolved_ts_gap")
            .unwrap_or_default();
        server_reader_utilization = M
            .get_metric_group("server_read_worker_busy_time")
            .unwrap_or_default();
        server_writer_utilization = M
            .get_metric_group("server_write_worker_busy_time")
            .unwrap_or_default();
        server_error_count = M.get_metric_group("server_error_count").unwrap_or_default();
        server_read_schedule_wait = M
            .get_metric_group("server_read_schedule_wait")
            .unwrap_or_default();
        server_write_schedule_wait = M
            .get_metric_group("server_write_schedule_wait")
            .unwrap_or_default();
        server_rest_time_to_handle_read = M
            .get_metric_group("server_rest_time_to_handle_read")
            .unwrap_or_default();
    } else {
        server_read_queue_length = M
            .get_metric_group("server_read_queue_length")
            .unwrap()
            .group_by_label(0, AggregateMethod::Max);
        server_write_queue_length = M
            .get_metric_group("server_write_queue_length")
            .unwrap()
            .group_by_label(0, AggregateMethod::Max);
        server_max_resolved_ts_gap = M
            .get_metric_group("server_max_resolved_ts_gap")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
        server_reader_utilization = M
            .get_metric_group("server_read_worker_busy_time")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
        server_writer_utilization = M
            .get_metric_group("server_write_worker_busy_time")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
        server_error_count = M
            .get_metric_group("server_error_count")
            .unwrap_or_default()
            .group_by_label(1, AggregateMethod::Max);
        server_read_schedule_wait = M
            .get_metric_group("server_read_schedule_wait")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
        server_write_schedule_wait = M
            .get_metric_group("server_write_schedule_wait")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
        server_rest_time_to_handle_read = M
            .get_metric_group("server_rest_time_to_handle_read")
            .unwrap_or_default()
            .group_by_label(0, AggregateMethod::Max);
    };
    // FIXME: aggregate on zone and state
    server_read_req_count = M
        .get_metric_group("server_read_request_count")
        .unwrap_or_default()
        .group_by_label(0, AggregateMethod::Min);
    server_write_req_count = M
        .get_metric_group("server_write_request_count")
        .unwrap_or_default()
        .group_by_label(0, AggregateMethod::Min);
    advance_resolved_ts_fail_count = M
        .get_metric_group("advance_resolved_ts_failure")
        .unwrap_or_default()
        .group_by_label(1, AggregateMethod::Max);

    plot_chart(
        &children_area,
        &font,
        colors,
        gauge_opts,
        &mut chart_id,
        "read queue length",
        server_read_queue_length,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        gauge_opts,
        &mut chart_id,
        "write queue length",
        server_write_queue_length,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        gauge_opts,
        &mut chart_id,
        "max resolved ts gap",
        server_max_resolved_ts_gap,
        ms_unit,
        ms_label,
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        counter_opts,
        &mut chart_id,
        "advance resolved ts fail count",
        advance_resolved_ts_fail_count,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        gauge_opts,
        &mut chart_id,
        "read worker utilization",
        server_reader_utilization,
        cfg.metrics_interval / 100,
        "%",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        gauge_opts,
        &mut chart_id,
        "write worker utilization",
        server_writer_utilization,
        cfg.metrics_interval / 100,
        "%",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        counter_opts,
        &mut chart_id,
        "read req count",
        server_read_req_count,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        counter_opts,
        &mut chart_id,
        "write req count",
        server_write_req_count,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        counter_opts,
        &mut chart_id,
        "error count",
        server_error_count,
        1,
        "",
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "read schedule wait",
        server_read_schedule_wait,
        ms_unit,
        ms_label,
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "write schedule wait",
        server_write_schedule_wait,
        ms_unit,
        ms_label,
    )?;
    plot_chart(
        &children_area,
        &font,
        colors,
        hist_latency_opts,
        &mut chart_id,
        "rest time to handle read",
        server_rest_time_to_handle_read,
        ms_unit,
        ms_label,
    )?;

    root.present()?;
    Ok(())
}

fn plot_chart(
    children_area: &[DrawingArea<SVGBackend, Shift>],
    font: &FontDesc,
    colors: [RGBColor; 30],
    opts: Opts,
    chart_id: &mut usize,
    chart_name: &str,
    metrics: MetricGroup,
    y_unit: Time,
    y_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let x_spec = 0f32..metrics.max_time() as f32;
    let y_spec = metrics.min_value() as f32..(if matches!(opts, Opts::Hist(HistOpts::Count)) {
        metrics.max_count() as i64
    } else {
        metrics.max_value()
    }) as f32
        * 1.2
        / y_unit as f32;
    let mut chart = ChartBuilder::on(&children_area[*chart_id])
        .caption(chart_name, font.clone())
        .margin(30)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_spec, y_spec)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_label_formatter(&|x| format!("{}{}", x, y_label))
        .draw()?;

    let mut data = metrics.data_point_series(opts);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (i, (name, series)) in data.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                series
                    .iter()
                    .map(|(x, y)| (*x as f32, *y as f32 / y_unit as f32)),
                colors[i],
            ))?
            .label(name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], colors[i].stroke_width(3))
            });
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.5))
        .draw()?;

    *chart_id += 1;
    Ok(())
}

struct Model {
    // model
    events: Events,
    servers: Vec<Server>,
    clients: Vec<Client>,
    app: App,

    // config
    num_replica: usize,
    metrics_interval: Time,
}

impl Model {
    #[allow(unused)]
    fn inject_crashed_client(&mut self) {
        let region_id = 1;
        println!(
            "inject lock in region {} in server {}",
            region_id,
            Self::find_leader_by_id(&mut self.servers, region_id).server_id
        );
        let mut events = self.events.borrow_mut();
        let start_ts = 200 * SECOND;
        events.push(Event::new(
            start_ts,
            EventType::PrewriteRequest,
            Box::new(move |model| {
                let client = Model::rand_client_in_zone(model, Zone::AZ1);
                // possible start_ts conflict, but ok
                let req = Request::new(
                    start_ts,
                    None,
                    EventType::PrewriteRequest,
                    10 * MILLISECOND,
                    client.id,
                    region_id,
                    false,
                );
                client.on_req(req);
            }),
        ));
        events.push(Event::new(
            400 * SECOND,
            EventType::CommitRequest,
            Box::new(move |model| {
                let client = Model::rand_client_in_zone(model, Zone::AZ1);
                // possible start_ts conflict, but ok
                let req = Request::new(
                    start_ts,
                    None,
                    EventType::CommitRequest,
                    10 * MILLISECOND,
                    client.id,
                    region_id,
                    false,
                );
                client.on_req(req);
            }),
        ));
    }

    #[allow(unused)]
    fn inject_app_retry(&mut self) {
        self.events.borrow_mut().push(Event::new(
            400 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.app.retry = true;
            }),
        ));
        self.events.borrow_mut().push(Event::new(
            600 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.app.retry = false;
            }),
        ));
    }

    #[allow(unused)]
    fn inject_read_delay(&mut self) {
        self.events.borrow_mut().push(Event::new(
            100 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.servers[0].read_delay = SECOND;
            }),
        ));
        self.events.borrow_mut().push(Event::new(
            150 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.servers[0].read_delay = 0;
            }),
        ));
    }

    fn inject_high_pressure(&mut self) {
        self.events.borrow_mut().push(Event::new(
            100 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.app.rate_exp_dist = Exp::new(model.app.txn_rate * 1.25).unwrap();
                // model.app.txn_interval_adjust_value = -(200 * MICROSECOND as i64);
            }),
        ));
        self.events.borrow_mut().push(Event::new(
            150 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.app.rate_exp_dist = Exp::new(model.app.txn_rate).unwrap();
                // model.app.txn_interval_adjust_value = 0;
            }),
        ));
    }

    fn init(&mut self, cfg: &Config) {
        // create regions

        // peer distribution rules:
        // region i's leader is in server i % num_servers. So leaders a balanced over servers.
        // 2 followers are in the other AZs,
        // try to make each server have the same number of followers,
        // by assigning a follower to a server who has fewest followers each time.

        assert!(self.servers.len() >= self.num_replica);
        let mut leader_idx = 0;
        assert!(self.servers[0].zone == Zone::AZ1);
        for region_id in 0..self.app.num_region {
            // leader in server[leader_idx]
            let mut leader = Peer {
                role: Role::Leader,
                server_id: self.servers[leader_idx].server_id,
                region_id,
                resolved_ts: 0,
                safe_ts: 0,
                lock_cf: HashSet::new(),
                advance_interval: cfg.server_config.advance_interval,
                broadcast_interval: cfg.server_config.broadcast_interval,
            };
            leader.update_resolved_ts(self.events.clone(), None);
            leader.broadcast_safe_ts(self.events.clone());
            self.servers[leader_idx].peers.insert(region_id, leader);

            let leader_zone = self.servers[leader_idx].zone.into_id();
            let follower_zones = (0..self.num_replica as u64).filter(|zone| zone != &leader_zone);
            for zone in follower_zones {
                let server_with_min_followers =
                    Self::fewest_follower_server_in_zone(&mut self.servers, Zone::from_id(zone));
                let server_id = server_with_min_followers.server_id;
                server_with_min_followers.peers.insert(
                    region_id,
                    Peer {
                        role: Role::Follower,
                        server_id,
                        region_id,
                        resolved_ts: 0,
                        safe_ts: 0,
                        lock_cf: HashSet::new(),
                        advance_interval: cfg.server_config.advance_interval,
                        broadcast_interval: cfg.server_config.broadcast_interval,
                    },
                );
            }
            leader_idx = (leader_idx + 1) % self.servers.len();
        }

        // start
        self.app.gen_txn();
        self.collect_metrics();
    }

    fn find_leader_by_id(servers: &mut [Server], region_id: u64) -> &mut Peer {
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

    fn find_followers_by_id(servers: &mut [Server], region_id: u64) -> Vec<&mut Peer> {
        servers
            .iter_mut()
            .filter_map(|s| {
                s.peers
                    .get_mut(&region_id)
                    .filter(|p| p.role == Role::Follower)
            })
            .collect()
    }

    fn find_server_by_id(servers: &mut [Server], server_id: u64) -> &mut Server {
        servers
            .iter_mut()
            .find(|s| s.server_id == server_id)
            .unwrap()
    }

    fn find_client_by_id(&mut self, client_id: u64) -> &mut Client {
        self.clients.iter_mut().find(|c| c.id == client_id).unwrap()
    }

    fn rand_client_in_zone(&mut self, zone: Zone) -> &mut Client {
        self.clients
            .iter_mut()
            .filter(|c| c.zone == zone)
            .choose(&mut rand::thread_rng())
            .unwrap()
    }

    fn rand_server_in_zone(servers: &mut [Server], zone: Zone) -> &mut Server {
        servers
            .iter_mut()
            .filter(|s| s.zone == zone)
            .choose(&mut rand::thread_rng())
            .unwrap()
    }

    fn fewest_follower_server_in_zone(servers: &mut [Server], zone: Zone) -> &mut Server {
        servers
            .iter_mut()
            .filter(|s| s.zone == zone)
            .min_by_key(|s| {
                s.peers
                    .iter()
                    .filter(|(_, p)| p.role == Role::Follower)
                    .count()
            })
            .unwrap()
    }

    fn collect_metrics(&mut self) {
        for server in &mut self.servers {
            M.set_gauge(
                "server_read_queue_length",
                vec![server.zone.to_string(), server.server_id.to_string()],
                server.read_task_queue.len() as i64,
            );
            M.set_gauge(
                "server_write_queue_length",
                vec![server.zone.to_string(), server.server_id.to_string()],
                server.write_task_queue.len() as i64,
            );

            let mut max_gap = 0;
            for peer in server.peers.values() {
                if peer.role == Role::Leader {
                    max_gap = max_gap.max(now() - peer.resolved_ts);
                }
            }
            M.set_gauge(
                "server_max_resolved_ts_gap",
                vec![server.zone.to_string(), server.server_id.to_string()],
                max_gap as i64,
            );

            let mut t = server.read_worker_time;
            for (start_time, _) in &mut server.read_workers.iter_mut().flatten() {
                t += now() - *start_time;
                *start_time = now();
            }
            server.read_worker_time = 0;
            M.set_gauge(
                "server_read_worker_busy_time",
                vec![server.zone.to_string(), server.server_id.to_string()],
                t as i64,
            );

            let mut t = server.write_worker_time;
            for (start_time, _) in &mut server.write_workers.iter_mut().flatten() {
                t += now() - *start_time;
                *start_time = now();
            }
            server.write_worker_time = 0;
            M.set_gauge(
                "server_write_worker_busy_time",
                vec![server.zone.to_string(), server.server_id.to_string()],
                t as i64,
            );
        }

        self.events.borrow_mut().push(Event::new(
            now() + self.metrics_interval,
            EventType::CollectMetrics,
            Box::new(move |model: &mut Model| {
                model.collect_metrics();
            }),
        ));
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Zone {
    AZ1,
    AZ2,
    AZ3,
}

impl Zone {
    fn into_id(self) -> u64 {
        match self {
            Zone::AZ1 => 0,
            Zone::AZ2 => 1,
            Zone::AZ3 => 2,
        }
    }

    fn from_id(id: u64) -> Self {
        match id % 3 {
            0 => Zone::AZ1,
            1 => Zone::AZ2,
            2 => Zone::AZ3,
            _ => panic!("invalid zone"),
        }
    }

    fn rand_zone() -> Self {
        Self::from_id(rand::random::<u64>() % 3)
    }
}

impl Display for Zone {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Zone::AZ1 => write!(f, "AZ1"),
            Zone::AZ2 => write!(f, "AZ2"),
            Zone::AZ3 => write!(f, "AZ3"),
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
    fn update_resolved_ts(&mut self, events: Events, min_memory_lock: Option<u64>) {
        assert!(self.role == Role::Leader);
        let min_lock_in_lock_cf = self.lock_cf.iter().min().copied().unwrap_or(u64::MAX);
        let candidate = now()
            .min(min_lock_in_lock_cf)
            .min(min_memory_lock.unwrap_or(u64::MAX));
        if candidate <= self.resolved_ts && now() > 0 {
            assert!(
                self.resolved_ts == 0
                    || (candidate == min_lock_in_lock_cf || candidate == min_memory_lock.unwrap())
            );
            if candidate == min_lock_in_lock_cf {
                M.inc_counter(
                    "advance_resolved_ts_failure",
                    vec!["lock".to_string(), self.server_id.to_string()],
                    1,
                );
            } else if candidate == min_memory_lock.unwrap() {
                M.inc_counter(
                    "advance_resolved_ts_failure",
                    vec!["memory_lock".to_string(), self.server_id.to_string()],
                    1,
                );
            } else {
                panic!();
            }
        }
        self.resolved_ts = max(self.resolved_ts, candidate);
        self.safe_ts = self.resolved_ts;

        let this_region_id = self.region_id;
        events.borrow_mut().push(Event::new(
            now() + self.advance_interval,
            EventType::ResolvedTsUpdate,
            Box::new(move |model: &mut Model| {
                let this_server_id =
                    Model::find_leader_by_id(&mut model.servers, this_region_id).server_id;
                let this_server = Model::find_server_by_id(&mut model.servers, this_server_id);
                let min_memory_lock = this_server.concurrency_manager.iter().min();
                let this = this_server.peers.get_mut(&this_region_id).unwrap();
                this.update_resolved_ts(model.events.clone(), min_memory_lock.copied());
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
    write_task_queue: VecDeque<(Time, Request)>,
    // Vec<(start handle time, task)>, the start time can be reset when collecting metrics. this is **only** used to calculate utilization
    read_workers: Vec<Option<(Time, Request)>>,
    write_workers: Vec<Option<(Time, Request)>>,
    // holding transactions during prewrite, and block resolved-ts
    concurrency_manager: HashSet<u64>,

    // config
    // a read request will abort at this time if it cannot finish in time
    read_timeout: Time,
    enable_async_commit: bool,
    read_delay: Time,

    // metrics
    // total busy time, in each metrics interval
    read_worker_time: Time,
    write_worker_time: Time,
}

impl Server {
    fn new(zone: Zone, events: Events, cfg: &Config) -> Self {
        Self {
            server_id: SERVER_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            peers: HashMap::new(),
            zone,
            events,
            read_task_queue: VecDeque::new(),
            read_workers: vec![None; cfg.server_config.num_read_workers],
            write_task_queue: VecDeque::new(),
            write_workers: vec![None; cfg.server_config.num_write_workers],
            concurrency_manager: Default::default(),
            read_timeout: cfg.server_config.read_timeout,
            enable_async_commit: cfg.server_config.enable_async_commit,
            read_delay: 0,
            read_worker_time: 0,
            write_worker_time: 0,
        }
    }

    fn on_req(&mut self, mut task: Request) {
        task.trace.record(format!(
            "{}: server-{}, recv req {}",
            now().pretty_print(),
            self.server_id,
            task.req_id
        ));
        match task.req_type {
            EventType::ReadRequest => {
                M.inc_counter(
                    "server_read_request_count",
                    vec![
                        self.zone.to_string(),
                        self.server_id.to_string(),
                        task.selector_state.to_string(),
                    ],
                    1,
                );
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
                M.inc_counter(
                    "server_write_request_count",
                    vec![
                        self.zone.to_string(),
                        self.server_id.to_string(),
                        task.req_type.to_string(),
                    ],
                    1,
                );
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
        M.record_hist(
            "server_read_schedule_wait",
            vec![self.zone.to_string(), self.server_id.to_string()],
            now() - accept_time,
        );

        let peer = self.peers.get_mut(&req.region_id).unwrap();

        let task_size = req.size + self.read_delay;
        let stale_read_ts = req.stale_read_ts;
        let is_retry = req.selector_state.is_retry();
        self.read_workers[worker_id] = Some((now(), req));
        let this_server_id = self.server_id;

        // safe ts check
        if let Some(stale_read_ts) = stale_read_ts {
            if stale_read_ts > peer.safe_ts {
                // schedule next task in queue
                let (_, task) = self.read_workers[worker_id].take().unwrap();
                if let Some((accept_time, task)) = self.read_task_queue.pop_front() {
                    self.handle_read(worker_id, task, accept_time);
                }

                M.inc_counter(
                    "server_error_count",
                    vec![
                        Error::DataIsNotReady.to_string(),
                        self.zone.to_string(),
                        self.server_id.to_string(),
                    ],
                    1,
                );

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

        // When the req gets out of the queue, how much time do I have to handle it before it times out?
        M.record_hist(
            "server_rest_time_to_handle_read",
            vec![self.zone.to_string(), this_server_id.to_string()],
            (accept_time + self.read_timeout).saturating_sub(now()),
        );

        // timeout check
        if !is_retry && accept_time + self.read_timeout < now() + task_size {
            // will timeout. It tries for until timeout, and then decide to abort.
            self.events.borrow_mut().push(Event::new(
                max(accept_time + self.read_timeout, now()),
                EventType::ReadRequestTimeout,
                Box::new(move |model: &mut Model| {
                    let this = Model::find_server_by_id(&mut model.servers, this_server_id);
                    M.inc_counter(
                        "server_error_count",
                        vec![
                            Error::ReadTimeout.to_string(),
                            this.zone.to_string(),
                            this.server_id.to_string(),
                        ],
                        1,
                    );

                    // schedule next task in queue
                    let (start_time, task) = this.read_workers[worker_id].take().unwrap();
                    this.read_worker_time += now() - start_time;
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
                let (start_time, task) = this.read_workers[worker_id].take().unwrap();
                this.read_worker_time += now() - start_time;
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
        M.record_hist(
            "server_write_schedule_wait",
            vec![self.zone.to_string(), self.server_id.to_string()],
            now() - accept_time,
        );
        let req_size = req.size;
        if req.req_type == EventType::PrewriteRequest {
            peer.lock_cf.insert(req.start_ts);
            if self.enable_async_commit {
                // in our model, a txn only writes only to 1 region.
                assert!(self.concurrency_manager.insert(req.start_ts));
            }
        }
        self.write_workers[worker_id] = Some((now(), req));
        let this_server_id = self.server_id;

        self.events.borrow_mut().push(Event::new(
            now() + req_size,
            EventType::HandleWrite,
            Box::new(move |model: &mut Model| {
                let this: &mut Server =
                    Model::find_server_by_id(&mut model.servers, this_server_id);
                let (start_time, task) = this.write_workers[worker_id].take().unwrap();
                this.write_worker_time += now() - start_time;
                let peer = this.peers.get_mut(&task.region_id).unwrap();
                assert!(peer.role == Role::Leader);

                match task.req_type {
                    EventType::PrewriteRequest => {
                        if this.enable_async_commit {
                            assert!(this.concurrency_manager.remove(&task.start_ts));
                        }
                    }
                    EventType::CommitRequest => {
                        assert!(peer.lock_cf.remove(&task.start_ts));
                    }
                    _ => {}
                }

                // schedule next task
                if let Some((accept_time, task)) = this.write_task_queue.pop_front() {
                    this.handle_write(worker_id, task, accept_time);
                }
                let this_zone = this.zone;

                // return resp
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

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum PeerSelectorState {
    // the default value of requests, should never use it
    Unknown,
    StaleRead(StaleReaderState),
    NormalRead(NormalReaderState),
    Write(WriterState),
}

impl PeerSelectorState {
    fn is_retry(&self) -> bool {
        match self {
            PeerSelectorState::Unknown => unreachable!(),
            PeerSelectorState::StaleRead(s) => match s {
                StaleReaderState::LocalStale => false,
                StaleReaderState::LeaderNormal => true,
                StaleReaderState::RandomFollowerNormal => true,
            },
            PeerSelectorState::NormalRead(s) => match s {
                NormalReaderState::Local => false,
                NormalReaderState::LeaderNormal => true,
                NormalReaderState::RandomFollowerNormal => true,
            },
            PeerSelectorState::Write(s) => match s {
                WriterState::Leader => false,
                WriterState::LeaderFailed => unreachable!(),
            },
        }
    }
}

impl Display for PeerSelectorState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerSelectorState::Unknown => write!(f, "Unknown"),
            PeerSelectorState::StaleRead(state) => write!(f, "StaleRead-{}", state),
            PeerSelectorState::NormalRead(state) => write!(f, "NormalRead-{}", state),
            PeerSelectorState::Write(state) => write!(f, "Write-{}", state),
        }
    }
}

// local(stale) -> leader(normal) -> random follower(normal) -> error
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum StaleReaderState {
    LocalStale,
    LeaderNormal,
    RandomFollowerNormal,
}

impl Display for StaleReaderState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StaleReaderState::LocalStale => write!(f, "LocalStale"),
            StaleReaderState::LeaderNormal => write!(f, "LeaderNormal"),
            StaleReaderState::RandomFollowerNormal => write!(f, "RandomFollowerNormal"),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum NormalReaderState {
    Local,
    LeaderNormal,
    RandomFollowerNormal,
}

impl Display for NormalReaderState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NormalReaderState::Local => write!(f, "Local"),
            NormalReaderState::LeaderNormal => write!(f, "LeaderNormal"),
            NormalReaderState::RandomFollowerNormal => write!(f, "RandomFollowerNormal"),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum WriterState {
    Leader,
    LeaderFailed,
}

impl Display for WriterState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WriterState::Leader => write!(f, "Leader"),
            WriterState::LeaderFailed => write!(f, "LeaderFailed"),
        }
    }
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
            state,
            server_ids_tried_for_normal_read: HashSet::new(),
            local_zone: local,
        }
    }

    fn next<'a>(&mut self, servers: &'a mut [Server], req: &mut Request) -> Option<&'a mut Server> {
        match &mut self.state {
            PeerSelectorState::StaleRead(state) => match state {
                StaleReaderState::LocalStale => {
                    assert_eq!(req.req_type, EventType::ReadRequest);
                    assert!(req.stale_read_ts.is_some());
                    *state = StaleReaderState::LeaderNormal;
                    let s = servers
                        .iter_mut()
                        .filter(|s| s.zone == self.local_zone)
                        .find(|s| s.peers.contains_key(&req.region_id))
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
                    let s = servers
                        .iter_mut()
                        .filter(|s| s.zone == self.local_zone)
                        .find(|s| s.peers.contains_key(&req.region_id))
                        .unwrap();
                    self.server_ids_tried_for_normal_read.insert(s.server_id);
                    self.state = PeerSelectorState::NormalRead(NormalReaderState::LeaderNormal);
                    Some(s)
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
            PeerSelectorState::Unknown => {
                unreachable!()
            }
        }
    }
}

struct Client {
    zone: Zone,
    id: u64,
    events: Events,
    // req_id -> (start_time, replica selector)
    pending_tasks: HashMap<u64, (Time, Rc<RefCell<PeerSelector>>)>,

    // config
    max_execution_time: Time,
}

impl Client {
    fn new(zone: Zone, events: Events, cfg: &Config) -> Self {
        Self {
            id: CLIENT_COUNTER.fetch_add(1, atomic::Ordering::SeqCst),
            zone,
            events,
            pending_tasks: HashMap::new(),
            max_execution_time: cfg.client_config.max_execution_time,
        }
    }

    // app sends a req to client
    fn on_req(&mut self, mut req: Request) {
        req.client_id = self.id;
        req.trace.record(format!(
            "{}: client {}, received req {}",
            now().pretty_print(),
            self.id,
            req.req_id,
        ));
        let selector = Rc::new(RefCell::new(PeerSelector::new(self.zone, &req)));
        self.issue_request(req, selector);
    }

    // send the req to the appropriate peer. If all peers have been tried, return error to app.
    fn issue_request(&mut self, mut req: Request, selector: Rc<RefCell<PeerSelector>>) {
        req.trace.record(format!(
            "{}: client {}, issued req {}, selector_state {}",
            now().pretty_print(),
            self.id,
            req.req_id,
            selector.borrow().state,
        ));
        let req_id = req.req_id;
        let mut req_clone = req.clone();
        let is_new_request = !self.pending_tasks.contains_key(&req.req_id);
        if is_new_request {
            self.pending_tasks
                .insert(req.req_id, (now(), selector.clone()));
        }
        // we should decide the target *now*, but to access the server list in the model, we decide when
        // the event the rpc is to be accepted by the server.
        let mut events = self.events.borrow_mut();
        let this_client_id = self.id;

        events.push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                let mut selector = selector.borrow_mut();
                req.selector_state = selector.state;
                let server = selector.next(&mut model.servers, &mut req);
                if let Some(server) = server {
                    req.trace.record(format!(
                        "{}: client {}, sending req {} to server {}",
                        now().pretty_print(),
                        this_client_id,
                        req.req_id,
                        server.server_id,
                    ));
                    server.on_req(req);
                } else {
                    // no server available, return error
                    let this = model.find_client_by_id(this_client_id);
                    this.pending_tasks.remove(&req_id).unwrap();
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

        if is_new_request {
            // check max_execution_timeout, if it did not get a response in time, abort and return error to app
            events.push(Event::new(
                now() + self.max_execution_time,
                EventType::CheckMaxExecutionTime,
                Box::new(move |model: &mut Model| {
                    let this = Model::find_client_by_id(model, this_client_id);
                    if let Some((start_time, _)) = this.pending_tasks.remove(&req_id) {
                        M.record_hist(
                            "kv_latency",
                            vec![
                                "fail".to_string(),
                                Error::MaxExecutionTimeExceeded.to_string(),
                                this.zone.to_string(),
                            ],
                            now() - start_time,
                        );
                        assert_eq!(
                            now() - start_time,
                            this.max_execution_time,
                            "now: {}, start_time: {}, max_execution_time: {}",
                            now().pretty_print(),
                            start_time.pretty_print(),
                            this.max_execution_time.pretty_print()
                        );

                        req_clone.trace.record(format!(
                            "{}: client {}, max execution time exceeded for req {}",
                            now().pretty_print(),
                            this.id,
                            req_id,
                        ));

                        this.events.borrow_mut().push(Event::new(
                            now() + rpc_latency(false),
                            EventType::AppResp,
                            Box::new(move |model: &mut Model| {
                                model
                                    .app
                                    .on_resp(req_clone, Some(Error::MaxExecutionTimeExceeded));
                            }),
                        ));
                    }
                }),
            ))
        }
    }

    fn on_resp(&mut self, mut req: Request, error: Option<Error>) {
        let entry = self.pending_tasks.get(&req.req_id);
        if entry.is_none() {
            // must be max execution time exceeded
            return;
        }
        let (start_time, selector) = entry.unwrap();

        if let Some(e) = error {
            req.trace.record(format!(
                "{}: client {}, received error {} for req {}",
                now().pretty_print(),
                self.id,
                e,
                req.req_id,
            ));
            M.record_hist(
                "kv_latency",
                vec!["fail".to_string(), e.to_string(), self.zone.to_string()],
                now() - start_time,
            );
            // retry other peers
            self.issue_request(req, selector.clone());
        } else {
            req.trace.record(format!(
                "{}: client {}, received success for req {}",
                now().pretty_print(),
                self.id,
                req.req_id,
            ));
            M.record_hist(
                "kv_latency",
                vec![
                    "ok".to_string(),
                    req.req_type.to_string(),
                    self.zone.to_string(),
                ],
                now() - start_time,
            );
            // success. respond to app
            self.pending_tasks.remove(&req.req_id).unwrap();
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
    read_staleness: Option<Time>,
    num_region: u64,

    // injection
    stop: bool,
    // add this time to the interval between two transactions
    txn_interval_adjust_value: i64,

    // configs
    txn_rate: f64,
    read_size_fn: Rc<dyn Fn() -> Time>,
    prewrite_size_fn: Rc<dyn Fn() -> Time>,
    commit_size_fn: Rc<dyn Fn() -> Time>,
    num_queries_fn: Rc<dyn Fn() -> u64>,
    read_only_ratio: f64,
    retry: bool,
    enable_trace: bool,

    // metrics
    txn_duration_stat: Histogram<Time>,
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

enum TransactionTrace {
    Enabled(Vec<String>),
    Disabled,
}

impl TransactionTrace {
    fn new(enabled: bool) -> Self {
        if enabled {
            Self::Enabled(Vec::new())
        } else {
            Self::Disabled
        }
    }

    fn record(&mut self, msg: String) {
        match self {
            Self::Enabled(messages) => {
                messages.push(msg);
            }
            Self::Disabled => {}
        }
    }

    fn extend(&mut self, msgs: Vec<String>) {
        match self {
            Self::Enabled(messages) => {
                messages.extend(msgs);
            }
            Self::Disabled => {}
        }
    }

    #[allow(unused)]
    fn dump(&self) {
        match self {
            Self::Enabled(messages) => {
                println!("===== transaction trace =====");
                for msg in messages {
                    println!("{}", msg);
                }
            }
            Self::Disabled => {}
        }
    }
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
    trace: Rc<RefCell<TransactionTrace>>,
}

impl Transaction {
    fn new(
        num_queries: u64,
        read_only: bool,
        read_staleness: Option<Time>,
        num_region: u64,
        read_size_fn: Rc<dyn Fn() -> Time>,
        prewrite_size_fn: Rc<dyn Fn() -> Time>,
        commit_size_fn: Rc<dyn Fn() -> Time>,
        enable_trace: bool,
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
                read_size_fn(),
                u64::MAX,
                rand::random::<u64>() % num_region,
                enable_trace,
            ));
        }
        let (mut prewrite_req, mut commit_req) = (None, None);
        if !read_only {
            let write_region = rand::random::<u64>() % num_region;
            prewrite_req = Some(Request::new(
                start_ts,
                None,
                EventType::PrewriteRequest,
                prewrite_size_fn(),
                u64::MAX,
                write_region,
                enable_trace,
            ));
            commit_req = Some(Request::new(
                start_ts,
                None,
                EventType::CommitRequest,
                commit_size_fn(),
                u64::MAX,
                write_region,
                enable_trace,
            ));
        }

        let trace = Rc::new(RefCell::new(TransactionTrace::new(enable_trace)));
        trace.borrow_mut().record(format!(
            "{}: app, txn created with start_ts {}",
            now().pretty_print(),
            start_ts.pretty_print()
        ));

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
            trace,
        }
    }
}

impl App {
    // req_rate: transactions per second
    fn new(events: Events, cfg: &Config) -> Self {
        Self {
            events,
            stop: false,
            rng: StdRng::from_seed(OsRng.gen()),
            rate_exp_dist: Exp::new(cfg.app_config.txn_rate).unwrap(),
            pending_transactions: HashMap::new(),
            txn_duration_stat: new_hist(),
            read_staleness: cfg.app_config.read_staleness,
            txn_rate: cfg.app_config.txn_rate,
            num_region: cfg.num_region,
            read_size_fn: cfg.app_config.read_size_fn.clone(),
            prewrite_size_fn: cfg.app_config.prewrite_size_fn.clone(),
            commit_size_fn: cfg.app_config.commit_size_fn.clone(),
            num_queries_fn: cfg.app_config.num_queries_fn.clone(),
            read_only_ratio: cfg.app_config.read_only_ratio,
            retry: cfg.app_config.retry,
            enable_trace: cfg.enable_trace,
            txn_interval_adjust_value: 0,
        }
    }

    fn gen_txn(&mut self) {
        // x% read-only, (1-x)% read-only transactions. independent of stale read
        M.inc_counter("app_gen_txn_counter", vec![], 1);
        let read_only = rand::random::<f64>() < self.read_only_ratio;
        let mut txn = Transaction::new(
            (self.num_queries_fn)(),
            read_only,
            if read_only { self.read_staleness } else { None },
            self.num_region,
            self.read_size_fn.clone(),
            self.prewrite_size_fn.clone(),
            self.commit_size_fn.clone(),
            self.enable_trace,
        );
        let zone = txn.zone;
        let req = txn.remaining_queries.pop_front().unwrap();
        let trace = txn.trace.clone();
        self.pending_transactions.insert(txn.start_ts, txn);
        self.issue_request(zone, req, trace);

        // Invariant: interval must be > 0, to avoid infinite loop.
        // And to make start_ts unique as start_ts is using now() directly
        let interval = ((self.rate_exp_dist.sample(&mut self.rng) * SECOND as f64) as Time).max(1);
        let trigger_time = ((now() as i64 + interval as i64)
            .saturating_add(self.txn_interval_adjust_value) as u64)
            .max(now() + 1);
        self.events.borrow_mut().push(Event::new(
            trigger_time,
            EventType::AppGen,
            Box::new(move |model: &mut Model| {
                if model.app.stop {
                    return;
                }
                model.app.gen_txn();
            }),
        ));
    }

    fn on_resp(&mut self, mut req: Request, error: Option<Error>) {
        let txn = self.pending_transactions.get_mut(&req.start_ts);
        if txn.is_none() {
            eprintln!(
                "txn not found: start_ts {} {}, could be an injected req",
                req.start_ts.pretty_print(),
                req.req_type
            );
            return;
        }
        let txn = txn.unwrap();
        txn.trace.borrow_mut().extend(req.trace.drain());

        if let Some(error) = error {
            if self.retry {
                // application retry immediately
                txn.trace.borrow_mut().record(format!(
                    "{}: app, retrying from app side, error: {}",
                    now().pretty_print(),
                    error
                ));
                let zone = txn.zone;
                M.inc_counter("app_retry", vec![zone.to_string()], 1);
                let trace = txn.trace.clone();
                let retry_req = Request::new(
                    req.start_ts,
                    req.stale_read_ts,
                    req.req_type,
                    req.size,
                    req.client_id,
                    req.region_id,
                    self.enable_trace,
                );
                trace.borrow_mut().record(format!(
                    "{}: app, retrying req {}-{} with new req {}",
                    now().pretty_print(),
                    req.req_type,
                    req.req_id,
                    retry_req.req_id
                ));
                self.issue_request(zone, retry_req, trace);
            } else {
                // application doesn't retry
                let txn = self.pending_transactions.remove(&req.start_ts).unwrap();
                txn.trace.borrow_mut().record(format!(
                    "{}: app, txn failed, error: {}",
                    now().pretty_print(),
                    error
                ));
                M.record_hist(
                    "app_txn_duration",
                    vec!["fail".to_owned(), txn.zone.to_string(), error.to_string()],
                    now() - txn.start_ts,
                );
            }
        } else {
            // success
            txn.trace.borrow_mut().record(format!(
                "{}: app, received ok response for req {}",
                now().pretty_print(),
                req.req_id,
            ));
            match txn.commit_phase {
                CommitPhase::ReadOnly => {
                    // no need to prewrite and commit. finish all read queries
                    txn.remaining_queries.pop_front();
                    if let Some(req) = txn.remaining_queries.pop_front() {
                        // send next query
                        let zone = txn.zone;
                        let trace = txn.trace.clone();
                        self.issue_request(zone, req, trace);
                    } else {
                        txn.commit_phase = CommitPhase::Committed;
                        M.record_hist(
                            "app_txn_duration",
                            vec!["ok".to_owned(), txn.zone.to_string()],
                            now() - txn.start_ts,
                        );
                        self.txn_duration_stat.record(now() - txn.start_ts).unwrap();
                        let txn = self.pending_transactions.remove(&req.start_ts).unwrap();
                        txn.trace.borrow().dump();
                    }
                }
                CommitPhase::NotYet => {
                    if let Some(req) = txn.remaining_queries.pop_front() {
                        // send next query
                        let zone = txn.zone;
                        let trace = txn.trace.clone();
                        self.issue_request(zone, req, trace);
                    } else {
                        txn.commit_phase = CommitPhase::Prewriting;
                        // send prewrite request
                        let zone = txn.zone;
                        let prewrite_req = txn.prewrite_req.take().unwrap();
                        let trace = txn.trace.clone();
                        self.issue_request(zone, prewrite_req, trace);
                    }
                }
                CommitPhase::Prewriting => {
                    txn.commit_phase = CommitPhase::Committing;
                    // send commit request
                    let zone = txn.zone;
                    txn.commit_ts = now();
                    let commit_req = txn.commit_req.take().unwrap();
                    let trace = txn.trace.clone();
                    self.issue_request(zone, commit_req, trace);
                }
                CommitPhase::Committing => {
                    txn.commit_phase = CommitPhase::Committed;
                    M.record_hist(
                        "app_txn_duration",
                        vec!["ok".to_owned(), txn.zone.to_string()],
                        now() - txn.start_ts,
                    );
                    self.txn_duration_stat.record(now() - txn.start_ts).unwrap();
                    let txn = self.pending_transactions.remove(&req.start_ts).unwrap();
                    txn.trace.borrow().dump();
                }
                CommitPhase::Committed => {
                    unreachable!();
                }
            }
        }
    }

    fn issue_request(&mut self, zone: Zone, req: Request, trace: Rc<RefCell<TransactionTrace>>) {
        trace.borrow_mut().record(format!(
            "{}: app, sending req {}-{} to zone {}",
            now().pretty_print(),
            req.req_type,
            req.req_id,
            zone,
        ));
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                model.rand_client_in_zone(zone).on_req(req);
            }),
        ));
    }
}

#[derive(Clone)]
enum RequestTrace {
    Enabled(Vec<String>),
    Disabled,
}

impl RequestTrace {
    fn new(enabled: bool) -> Self {
        if enabled {
            Self::Enabled(Vec::new())
        } else {
            Self::Disabled
        }
    }

    fn record(&mut self, msg: String) {
        match self {
            Self::Enabled(messages) => {
                messages.push(msg);
            }
            Self::Disabled => {}
        }
    }

    fn drain(&mut self) -> Vec<String> {
        match self {
            Self::Enabled(messages) => std::mem::take(messages),
            Self::Disabled => Vec::new(),
        }
    }
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

    // metrics
    selector_state: PeerSelectorState,
    trace: RequestTrace,
}

impl Request {
    fn new(
        start_ts: u64,
        stale_read_ts: Option<u64>,
        req_type: EventType,
        size: Time,
        client_id: u64,
        region: u64,
        enable_trace: bool,
    ) -> Self {
        let req_id = TASK_COUNTER.fetch_add(1, atomic::Ordering::SeqCst);
        let mut trace = RequestTrace::new(enable_trace);
        trace.record(format!(
            "{}: req {}-{} created",
            now().pretty_print(),
            req_type,
            req_id,
        ));
        Self {
            start_ts,
            stale_read_ts,
            req_type,
            req_id,
            client_id,
            size,
            region_id: region,
            selector_state: PeerSelectorState::Unknown,
            trace,
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

    fn pop(&mut self) -> Option<Event> {
        self.events.pop()
    }

    fn push(&mut self, event: Event) {
        assert!(
            now() <= event.trigger_time,
            "now {}, {} {}",
            now(),
            event.trigger_time,
            event.event_type
        );
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    CollectMetrics,
    CheckMaxExecutionTime,
    #[allow(unused)]
    Injection,
}

impl Display for EventType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::HandleRead => write!(f, "HandleRead"),
            EventType::HandleWrite => write!(f, "HandleWrite"),
            EventType::ResolvedTsUpdate => write!(f, "ResolvedTsUpdate"),
            EventType::BroadcastSafeTs => write!(f, "BroadcastSafeTs"),
            EventType::ReadRequest => write!(f, "ReadRequest"),
            EventType::ReadRequestTimeout => write!(f, "ReadRequestTimeout"),
            EventType::PrewriteRequest => write!(f, "PrewriteRequest"),
            EventType::CommitRequest => write!(f, "CommitRequest"),
            EventType::Response => write!(f, "Response"),
            EventType::AppResp => write!(f, "AppResp"),
            EventType::AppGen => write!(f, "AppGen"),
            EventType::CollectMetrics => write!(f, "CollectMetrics"),
            EventType::CheckMaxExecutionTime => write!(f, "CheckMaxExecutionTime"),
            EventType::Injection => write!(f, "Injection"),
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
enum Error {
    // server is busy - deadline exceeded
    ReadTimeout,
    // all servers are unavailable
    RegionUnavailable,
    DataIsNotReady,
    MaxExecutionTimeExceeded,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ReadTimeout => write!(f, "ReadTimeout"),
            Error::RegionUnavailable => write!(f, "RegionUnavailable"),
            Error::DataIsNotReady => write!(f, "DataIsNotReady"),
            Error::MaxExecutionTimeExceeded => write!(f, "MaxExecutionTimeExceeded"),
        }
    }
}

fn new_hist() -> Histogram<Time> {
    Histogram::<Time>::new(3).unwrap()
}
