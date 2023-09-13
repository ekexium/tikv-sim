use hdrhistogram::Histogram;
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::cmp::{max, min, Ordering};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
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
struct Config {
    num_replica: usize,
    num_region: u64,
    max_time: Time,
    metrics_interval: Time,
    server_config: ServerConfig,
    app_config: AppConfig,
}

#[derive(Clone)]
struct ServerConfig {
    num_read_workers: usize,
    num_write_workers: usize,
    read_timeout: Time,
    advance_interval: Time,
    broadcast_interval: Time,
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
        num_region: 100,
        max_time: 900 * SECOND,
        metrics_interval: SECOND,
        server_config: ServerConfig {
            num_read_workers: 10,
            num_write_workers: 10,
            read_timeout: SECOND,
            advance_interval: 5 * SECOND,
            broadcast_interval: 5 * SECOND,
        },
        app_config: AppConfig {
            retry: false,
            txn_rate: 3200.0,
            read_staleness: Some(15 * SECOND),
            read_size_fn: Rc::new(|| (rand::random::<u64>() % 5 + 1) * MILLISECOND),
            prewrite_size_fn: Rc::new(|| (rand::random::<u64>() % 30 + 1) * MILLISECOND),
            commit_size_fn: Rc::new(|| (rand::random::<u64>() % 20 + 1) * MILLISECOND),
            num_queries_fn: Rc::new(|| 5),
            read_only_ratio: 0.95,
        },
    };
    // assert_eq!(config.metrics_interval, SECOND, "why bother");

    let mut model = Model {
        events: events.clone(),
        num_replica: config.num_replica,
        metrics_interval: config.metrics_interval,
        servers: vec![
            Server::new(Zone::AZ1, events.clone(), &config),
            Server::new(Zone::AZ2, events.clone(), &config),
            Server::new(Zone::AZ3, events.clone(), &config),
        ],
        clients: vec![
            Client::new(Zone::AZ1, events.clone()),
            Client::new(Zone::AZ2, events.clone()),
            Client::new(Zone::AZ3, events.clone()),
        ],
        app: App::new(events.clone(), &config),
        app_ok_transaction_durations: vec![],
        app_fail_transaction_durations: vec![],
        kv_ok_durations: Default::default(),
        kv_error_durations: Default::default(),
        server_max_resolved_ts_gap: vec![],
        server_read_queue_length: Default::default(),
        server_write_queue_length: Default::default(),
        advance_resolved_ts_failure: vec![],
        server_read_worker_busy_time: vec![],
        server_write_worker_busy_time: vec![],
        server_read_req_count: vec![],
        server_write_req_count: vec![],
    };
    model.init(&config);
    model.inject();

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
        CURRENT_TIME.store(event.trigger_time, atomic::Ordering::SeqCst);
        drop(events_mut);
        (event.f)(&mut model);
    }

    draw_metrics(&model, &config)?;
    Ok(())
}

fn draw_metrics(model: &Model, cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let num_graphs = 14usize + 3/* num_server*/;
    let root = SVGBackend::new("0.svg", (1200, num_graphs as u32 * 300)).into_drawing_area();
    let children_area = root.split_evenly(((num_graphs + 1) / 2, 2));
    let xs = (0..=model.app_ok_transaction_durations.len()).map(|i| i as f32);
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

    let mut chart_id = 0;
    // app_ok_transaction_rate
    {
        let txn_ok_rate = model.app_ok_transaction_durations.iter().map(|b| b.len());
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("successful transaction rate (per interval)", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.app_ok_transaction_durations.len() as f32,
                0f32..txn_ok_rate.clone().max().unwrap() as f32 * 1.2,
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            xs.clone().zip(txn_ok_rate.map(|x| x as f32)),
            &colors[0],
        ))?;
    }

    // app_ok_txn_latency
    {
        chart_id += 1;
        let (y_unit, y_label) = (MILLISECOND, "ms");
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("successful txn latency", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.app_ok_transaction_durations.len() as f32,
                0f32..(model
                    .app_ok_transaction_durations
                    .iter()
                    .map(|x| x.max())
                    .max()
                    .unwrap()
                    / y_unit) as f32
                    * 1.2,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;
        // mean latency
        chart
            .draw_series(LineSeries::new(
                xs.clone().zip(
                    model
                        .app_ok_transaction_durations
                        .iter()
                        .map(|x| (x.mean() / y_unit as f64) as f32),
                ),
                &colors[0],
            ))?
            .label("mean")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[0]));
        // p99 latency
        chart
            .draw_series(LineSeries::new(
                xs.clone().zip(
                    model
                        .app_ok_transaction_durations
                        .iter()
                        .map(|x| (x.value_at_quantile(0.99) / y_unit) as f32),
                ),
                &colors[1],
            ))?
            .label("p99")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[1]));
        // legend
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // app error rate
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("app error rate (per interval)", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.app_fail_transaction_durations.len() as f32,
                0f32..model
                    .app_fail_transaction_durations
                    .iter()
                    .map(|x| x.values().map(|x| x.len()).max().unwrap_or(0))
                    .max()
                    .unwrap_or(0) as f32
                    * 1.2,
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let errors = model
            .app_fail_transaction_durations
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, error) in errors.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .app_fail_transaction_durations
                            .iter()
                            .map(|x| x.get(error).map(|x| x.len()).unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(error.to_string())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // KV rate
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("kv ok query rate (per interval)", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.kv_ok_durations.len() as f32,
                0f32..model
                    .kv_ok_durations
                    .iter()
                    .map(|x| x.values().map(|x| x.len()).sum::<u64>())
                    .max()
                    .unwrap() as f32
                    * 1.2,
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let errors = model
            .kv_ok_durations
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, error) in errors.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .kv_ok_durations
                            .iter()
                            .map(|x| x.get(error).map(|x| x.len()).unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(error.to_string())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // kv error rates
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("kv error rates (per interval)", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.kv_error_durations.len() as f32,
                0f32..model
                    .kv_error_durations
                    .iter()
                    .map(|x| x.values().map(|x| x.len()).sum::<u64>())
                    .max()
                    .unwrap() as f32
                    * 1.2,
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let errors = model
            .kv_error_durations
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, error) in errors.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .kv_error_durations
                            .iter()
                            .map(|x| x.get(error).map(|x| x.len()).unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(error.to_string())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // KV OK latency
    {
        chart_id += 1;
        let (y_unit, y_label) = (MILLISECOND, "ms");
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("kv ok latency", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.kv_ok_durations.len() as f32,
                0f32..(model
                    .kv_ok_durations
                    .iter()
                    .map(|x| x.values().map(|x| x.max()).max().unwrap_or(0))
                    .max()
                    .unwrap()
                    / y_unit) as f32
                    * 1.2,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;

        let req_types = model
            .kv_ok_durations
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        // mean latency
        for (i, req_type) in req_types.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(model.kv_ok_durations.iter().map(|x| {
                        x.get(req_type)
                            .map(|x| (x.mean() / y_unit as f64) as f32)
                            .unwrap_or(0.0)
                    })),
                    colors[i],
                ))?
                .label(req_type.to_string() + "-mean")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        // p99 latency
        for (i, req_type) in req_types.iter().enumerate() {
            chart
                .draw_series(PointSeries::<(f32, f32), _, Circle<_, _>, f32>::new(
                    xs.clone().zip(model.kv_ok_durations.iter().map(|x| {
                        x.get(req_type)
                            .map(|x| (x.value_at_quantile(0.99) / y_unit) as f32)
                            .unwrap_or(0.0)
                    })),
                    3f32,
                    colors[i],
                ))?
                .label(req_type.to_string() + "-p99")
                .legend(move |(x, y)| Circle::new((x, y), 3f32, colors[i]));
        }

        // legend
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // KV error latency
    {
        chart_id += 1;
        let (y_unit, y_label) = (MILLISECOND, "ms");
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("kv error latency", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.kv_error_durations.len() as f32,
                0f32..(model
                    .kv_error_durations
                    .iter()
                    .map(|x| x.values().map(|x| x.max()).max().unwrap_or(0))
                    .max()
                    .unwrap()
                    / y_unit) as f32
                    * 1.2,
            )?;

        let errors = model
            .kv_error_durations
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;

        // mean latency
        for (i, error) in errors.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(model.kv_error_durations.iter().map(|x| {
                        x.get(error)
                            .map(|x| (x.mean() / y_unit as f64) as f32)
                            .unwrap_or(0.0)
                    })),
                    colors[i],
                ))?
                .label(error.to_string() + "-mean")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        // p99 latency
        for (i, error) in errors.iter().enumerate() {
            chart
                .draw_series(PointSeries::<(f32, f32), _, Circle<_, _>, f32>::new(
                    xs.clone().zip(model.kv_error_durations.iter().map(|x| {
                        x.get(error)
                            .map(|x| (x.value_at_quantile(0.99) / y_unit) as f32)
                            .unwrap_or(0.0)
                    })),
                    3f32,
                    colors[i],
                ))?
                .label(error.to_string() + "-p99")
                .legend(move |(x, y)| Circle::new((x, y), 3f32, colors[i]));
        }

        // legend
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // read queue length
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("read queue length", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.server_read_queue_length.len() as f32,
                0f32..(model
                    .server_read_queue_length
                    .iter()
                    .map(|x| x.values().copied().max().unwrap_or(0))
                    .max()
                    .unwrap() as f32
                    * 1.2)
                    .max(1.0),
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let server_ids = model
            .server_read_queue_length
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, server_id) in server_ids.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .server_read_queue_length
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // write queue length
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("write queue length", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.server_write_queue_length.len() as f32,
                0f32..(model
                    .server_write_queue_length
                    .iter()
                    .map(|x| x.values().copied().max().unwrap_or(0))
                    .max()
                    .unwrap() as f32
                    * 1.2)
                    .max(1.0),
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let server_ids = model
            .server_write_queue_length
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, server_id) in server_ids.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .server_write_queue_length
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // server_max_resolved_ts_gap
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("max resolved ts gap (per interval)", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .build_cartesian_2d(
                0f32..model.server_max_resolved_ts_gap.len() as f32,
                0f32..(model
                    .server_max_resolved_ts_gap
                    .iter()
                    .map(|x| x.values().copied().max().unwrap_or(0))
                    .max()
                    .unwrap() as f32
                    * 1.2
                    / SECOND as f32)
                    .max(1.0),
            )?;

        chart
            .configure_mesh()
            // .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}s", x))
            .y_labels(10)
            .draw()?;

        let server_ids = model
            .server_max_resolved_ts_gap
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, server_id) in server_ids.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .server_max_resolved_ts_gap
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32 / SECOND as f32),
                    ),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // advance_ts_fail_count
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("advance resolved ts failure count", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.advance_resolved_ts_failure.len() as f32,
                0f32..(*model
                    .advance_resolved_ts_failure
                    .iter()
                    .map(|x| x.values().max().unwrap_or(&0))
                    .max()
                    .unwrap_or(&0) as f32
                    * 1.2)
                    .max(1.0),
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let server_ids = model
            .advance_resolved_ts_failure
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, server_id) in server_ids.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .advance_resolved_ts_failure
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // read worker utilization
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption(
                "read worker utilization (busy time / total time)",
                font.clone(),
            )
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.server_read_worker_busy_time.len() as f32,
                0f32..cfg.server_config.num_read_workers as f32 * 1.2 * 100.0,
            )?;
        let servers: HashSet<u64> = model
            .server_read_worker_busy_time
            .iter()
            .flat_map(|x| x.keys())
            .copied()
            .collect();

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{}%", x))
            .draw()?;

        for (i, server_id) in servers.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone()
                        .zip(model.server_read_worker_busy_time.iter().map(|x| {
                            x.get(server_id).copied().unwrap_or(0) as f32
                                / cfg.metrics_interval as f32
                                * 100.0
                        })),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // write worker utilization
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption(
                "write worker utilization (busy time / total time)",
                font.clone(),
            )
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.server_write_worker_busy_time.len() as f32,
                0f32..cfg.server_config.num_write_workers as f32 * 1.2 * 100.0,
            )?;
        let servers: HashSet<u64> = model
            .server_write_worker_busy_time
            .iter()
            .flat_map(|x| x.keys())
            .copied()
            .collect();

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{}%", x))
            .draw()?;

        for (i, server_id) in servers.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone()
                        .zip(model.server_write_worker_busy_time.iter().map(|x| {
                            x.get(server_id).copied().unwrap_or(0) as f32
                                / cfg.metrics_interval as f32
                                * 100.0
                        })),
                    colors[i],
                ))?
                .label(format!("server-{}", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // server read req count
    {
        // draw graph for each server
        for server in &model.servers {
            chart_id += 1;
            let mut chart = ChartBuilder::on(&children_area[chart_id])
                .caption(
                    format!("server-{} read req count", server.server_id),
                    font.clone(),
                )
                .margin(30)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(
                    0f32..model.server_read_req_count.len() as f32,
                    0f32..(*model
                        .server_read_req_count
                        .iter()
                        .map(|x| {
                            x.get(&server.server_id)
                                .map(|x| x.values().max().unwrap_or(&0))
                                .unwrap_or(&0)
                        })
                        .max()
                        .unwrap_or(&0) as f32
                        * 1.2)
                        .max(1.0),
                )?;

            chart.configure_mesh().disable_mesh().draw()?;
            let states: HashSet<PeerSelectorState> = model
                .server_read_req_count
                .iter()
                .flat_map(|x| {
                    x.get(&server.server_id)
                        .map(|x| x.keys().copied().collect::<HashSet<_>>())
                        .unwrap_or_default()
                })
                .collect();
            let mut states: Vec<_> = states.into_iter().collect();
            states.sort();

            for (i, state) in states.iter().enumerate() {
                chart
                    .draw_series(LineSeries::new(
                        xs.clone().zip(model.server_read_req_count.iter().map(|x| {
                            x.get(&server.server_id)
                                .map(|x| x.get(state).copied().unwrap_or(0) as f32)
                                .unwrap_or(0.0)
                        })),
                        colors[i],
                    ))?
                    .label(state.to_string())
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
            }

            chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .background_style(WHITE.mix(0.5))
                .draw()?;
        }
    }

    // server write request count
    {
        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("server write req count", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..model.server_write_req_count.len() as f32,
                0f32..(*model
                    .server_write_req_count
                    .iter()
                    .map(|x| {
                        x.values()
                            .map(|x| x.values().max().unwrap_or(&0))
                            .max()
                            .unwrap_or(&0)
                    })
                    .max()
                    .unwrap_or(&0) as f32
                    * 1.2)
                    .max(1.0),
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let server_state_combinations: HashSet<(&u64, &PeerSelectorState)> = model
            .server_write_req_count
            .iter()
            .flat_map(|x| {
                x.iter().flat_map(|(server_id, states)| {
                    states.keys().map(move |state| (server_id, state))
                })
            })
            .collect();
        let mut server_state_combinations: Vec<_> = server_state_combinations.into_iter().collect();
        server_state_combinations.sort();

        for (i, (server_id, state)) in server_state_combinations.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(model.server_write_req_count.iter().map(|x| {
                        x.get(server_id)
                            .map(|x| x.get(state).copied().unwrap_or(0) as f32)
                            .unwrap_or(0.0)
                    })),
                    colors[i],
                ))?
                .label(format!("server-{}-{}", server_id, state))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    root.present()?;
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

    // metrics
    app_ok_transaction_durations: Vec<Histogram<Time>>,
    app_fail_transaction_durations: Vec<HashMap<Error, Histogram<Time>>>,
    kv_ok_durations: Vec<HashMap<EventType, Histogram<Time>>>,
    kv_error_durations: Vec<HashMap<Error, Histogram<Time>>>,
    // gauge, server_id -> max gap
    server_max_resolved_ts_gap: Vec<HashMap<u64, u64>>,
    // gauge, server_id -> length
    server_read_queue_length: Vec<HashMap<u64, u64>>,
    // gauge, server_id -> length
    server_write_queue_length: Vec<HashMap<u64, u64>>,
    // server id -> count
    advance_resolved_ts_failure: Vec<HashMap<u64, u64>>,
    // server id -> total time
    server_read_worker_busy_time: Vec<HashMap<u64, u64>>,
    // server id -> total time
    server_write_worker_busy_time: Vec<HashMap<u64, u64>>,
    server_read_req_count: Vec<HashMap<u64, HashMap<PeerSelectorState, u64>>>,
    server_write_req_count: Vec<HashMap<u64, HashMap<PeerSelectorState, u64>>>,
}

impl Model {
    fn inject(&mut self) {
        let region_id = rand::thread_rng().gen_range(0..self.app.num_region);
        let mut events = self.events.borrow_mut();
        let start_ts = 300 * SECOND;
        events.push(Event::new(
            300 * SECOND,
            EventType::PrewriteRequest,
            Box::new(move |model| {
                let client = Model::find_client_by_zone(model, Zone::AZ1);
                // possible start_ts conflict, but ok
                let req = Request::new(
                    start_ts,
                    None,
                    EventType::PrewriteRequest,
                    10 * MILLISECOND,
                    client.id,
                    region_id,
                );
                client.on_req(req);
            }),
        ));
        events.push(Event::new(
            700 * SECOND,
            EventType::CommitRequest,
            Box::new(move |model| {
                let client = Model::find_client_by_zone(model, Zone::AZ1);
                // possible start_ts conflict, but ok
                let req = Request::new(
                    start_ts,
                    None,
                    EventType::CommitRequest,
                    10 * MILLISECOND,
                    client.id,
                    region_id,
                );
                client.on_req(req);
            }),
        ));
    }

    fn init(&mut self, cfg: &Config) {
        // create regions
        assert!(self.servers.len() >= self.num_replica);
        let mut leader_idx = 0;
        for region_id in 0..self.app.num_region {
            // leader in server[leader_idx], `num_replica - 1` followers in server[leader_idx + 1]..server[leader_idx + num_replica - 1]
            let mut leader = Peer {
                role: Role::Leader,
                server_id: self.servers[leader_idx].server_id,
                region_id,
                resolved_ts: 0,
                safe_ts: 0,
                lock_cf: HashSet::new(),
                advance_interval: cfg.server_config.advance_interval,
                broadcast_interval: cfg.server_config.broadcast_interval,
                fail_advance_resolved_ts_stat: 0,
            };
            leader.update_resolved_ts(self.events.clone());
            leader.broadcast_safe_ts(self.events.clone());
            self.servers[leader_idx].peers.insert(region_id, leader);
            for follow_id in 1..=self.num_replica - 1 {
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
                        advance_interval: cfg.server_config.advance_interval,
                        broadcast_interval: cfg.server_config.broadcast_interval,
                        fail_advance_resolved_ts_stat: 0,
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

    fn find_client_by_zone(&mut self, zone: Zone) -> &mut Client {
        self.clients.iter_mut().find(|c| c.zone == zone).unwrap()
    }

    fn collect_metrics(&mut self) {
        {
            self.app_ok_transaction_durations
                .push(self.app.txn_duration_stat.clone());
            self.app.txn_duration_stat.reset();
        }

        {
            // max resolved ts gap
            let mut map = HashMap::new();
            for server in &self.servers {
                let mut max_gap = 0;
                for peer in server.peers.values() {
                    if peer.role == Role::Leader {
                        max_gap = max_gap.max(now() - peer.resolved_ts);
                    }
                }
                map.insert(server.server_id, max_gap);
            }
            self.server_max_resolved_ts_gap.push(map);
        }

        {
            let mut map = HashMap::new();
            for (error, stat) in &mut self.app.failed_txn_stat {
                map.insert(*error, stat.clone());
                stat.reset();
            }
            self.app_fail_transaction_durations.push(map);
        }

        {
            let mut map = HashMap::new();
            for client in &mut self.clients {
                for (req_type, stat) in &mut client.success_latency_stat {
                    map.entry(*req_type)
                        .or_insert_with(|| {
                            Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap()
                        })
                        .add(stat.clone())
                        .unwrap();
                    stat.reset()
                }
            }
            self.kv_ok_durations.push(map);
        }

        {
            let mut map: HashMap<Error, Histogram<Time>> = HashMap::new();
            for client in &mut self.clients {
                for (error, stat) in &mut client.error_latency_stat {
                    map.entry(*error)
                        .or_insert_with(|| {
                            Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3).unwrap()
                        })
                        .add(stat.clone())
                        .unwrap();
                    stat.reset();
                }
            }
            self.kv_error_durations.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &self.servers {
                map.insert(server.server_id, server.read_task_queue.len() as u64);
            }
            self.server_read_queue_length.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &self.servers {
                map.insert(server.server_id, server.write_task_queue.len() as u64);
            }
            self.server_write_queue_length.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                for peer in &mut server.peers.values_mut() {
                    if peer.role == Role::Leader {
                        *map.entry(server.server_id).or_insert(0) +=
                            peer.fail_advance_resolved_ts_stat;
                        peer.fail_advance_resolved_ts_stat = 0;
                    }
                }
            }
            self.advance_resolved_ts_failure.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                let mut t = server.read_worker_time;
                // calculate those in workers
                for worker in &mut server.read_workers {
                    if let Some((start_time, _)) = worker {
                        t += now() - *start_time;
                        *start_time = now();
                    }
                }
                server.read_worker_time = 0;
                map.insert(server.server_id, t);
            }
            self.server_read_worker_busy_time.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                let mut t = server.write_worker_time;
                // calculate those in workers
                for worker in &mut server.write_workers {
                    if let Some((start_time, _)) = worker {
                        t += now() - *start_time;
                        *start_time = now();
                    }
                }
                server.write_worker_time = 0;
                map.insert(server.server_id, t);
            }
            self.server_write_worker_busy_time.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                map.insert(server.server_id, server.read_req_count.drain().collect());
            }
            self.server_read_req_count.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                map.insert(server.server_id, server.write_req_count.drain().collect());
            }
            self.server_write_req_count.push(map);
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
    fn rand_zone() -> Self {
        match rand::random::<u64>() % 3 {
            0 => Zone::AZ1,
            1 => Zone::AZ2,
            2 => Zone::AZ3,
            _ => panic!("invalid zone"),
        }
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
    fail_advance_resolved_ts_stat: u64,
}

impl Peer {
    fn update_resolved_ts(&mut self, events: Events) {
        assert!(self.role == Role::Leader);
        let min_lock = *self.lock_cf.iter().min().unwrap_or(&u64::MAX);
        let new_resolved_ts = min(now(), min_lock);
        if new_resolved_ts <= self.resolved_ts && now() > 0 {
            assert!(self.resolved_ts == 0 || new_resolved_ts == min_lock);
            self.fail_advance_resolved_ts_stat += 1;
        }
        self.resolved_ts = max(self.resolved_ts, new_resolved_ts);
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
    write_task_queue: VecDeque<(Time, Request)>,
    // Vec<(start handle time, task)>, the start time can be reset when collecting metrics. this is **only** used to calculate utilization
    read_workers: Vec<Option<(Time, Request)>>,
    write_workers: Vec<Option<(Time, Request)>>,

    // a read request will abort at this time if it cannot finish in time
    read_timeout: Time,

    // metrics
    error_count: u64,
    write_schedule_wait_stat: Histogram<Time>,
    read_schedule_wait_stat: Histogram<Time>,
    // total busy time, in each metrics interval
    read_worker_time: Time,
    write_worker_time: Time,
    read_req_count: HashMap<PeerSelectorState, u64>,
    write_req_count: HashMap<PeerSelectorState, u64>,
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
            read_schedule_wait_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
            read_timeout: cfg.server_config.read_timeout,
            error_count: 0,
            write_schedule_wait_stat: Histogram::<Time>::new_with_bounds(
                NANOSECOND,
                60 * SECOND,
                3,
            )
            .unwrap(),
            read_worker_time: 0,
            write_worker_time: 0,
            read_req_count: HashMap::new(),
            write_req_count: HashMap::new(),
        }
    }

    fn on_req(&mut self, task: Request) {
        match task.req_type {
            EventType::ReadRequest => {
                *self.read_req_count.entry(task.selector_state).or_insert(0) += 1;
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
                *self.write_req_count.entry(task.selector_state).or_insert(0) += 1;
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
        self.write_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();
        let mut req_size = req.size;
        if req.req_type == EventType::PrewriteRequest {
            peer.lock_cf.insert(req.start_ts);
        }
        self.write_workers[worker_id] = Some((now(), req));
        let this_server_id = self.server_id;

        // if self.server_id == 0 {
        //     // 1% chance req_size += 1 second
        //     if rand::random::<u64>() % 100 == 0 {
        //         req_size += 1 * SECOND;
        //     }
        // }

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

                if task.req_type == EventType::CommitRequest {
                    assert!(peer.lock_cf.remove(&task.start_ts));
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
    latency_stat: Histogram<Time>,
    error_latency_stat: HashMap<Error, Histogram<Time>>,
    success_latency_stat: HashMap<EventType, Histogram<Time>>,
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
            success_latency_stat: HashMap::new(),
        }
    }

    // app sends a req to client
    fn on_req(&mut self, mut req: Request) {
        req.client_id = self.id;
        req.trace.messages.push(format!(
            "{}: client {}-{} received req {}",
            now().pretty_print(),
            self.zone,
            self.id,
            req.req_id,
        ));
        let selector = Rc::new(RefCell::new(PeerSelector::new(self.zone, &req)));
        self.issue_request(req, selector);
    }

    // send the req to the appropriate peer. If all peers have been tried, return error to app.
    fn issue_request(&mut self, mut req: Request, selector: Rc<RefCell<PeerSelector>>) {
        req.trace.messages.push(format!(
            "{}: client {}-{} issued req {}, selector_state {}",
            now().pretty_print(),
            self.zone,
            self.id,
            req.req_id,
            selector.borrow().state,
        ));
        self.pending_tasks
            .insert(req.req_id, (now(), selector.clone()));
        // we should decide the target *now*, but to access the server list in the model, we decide when
        // the event the rpc is to be accepted by the server.
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                let mut selector = selector.borrow_mut();
                req.selector_state = selector.state;
                let server = selector.next(&mut model.servers, &mut req);
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

    fn on_resp(&mut self, mut req: Request, error: Option<Error>) {
        let (start_time, selector) = self.pending_tasks.get(&req.req_id).unwrap();
        self.latency_stat.record(now() - start_time).unwrap();

        if let Some(e) = error {
            req.trace.messages.push(format!(
                "{}: client {}-{} received error {} for req {}",
                now().pretty_print(),
                self.zone,
                self.id,
                e,
                req.req_id,
            ));
            self.error_latency_stat
                .entry(e)
                .or_insert_with(|| Histogram::<Time>::new_with_bounds(1, 60 * SECOND, 3).unwrap())
                .record(now() - start_time)
                .unwrap();
            // retry other peers
            self.issue_request(req, selector.clone());
        } else {
            req.trace.messages.push(format!(
                "{}: client {}-{} received success for req {}",
                now().pretty_print(),
                self.zone,
                self.id,
                req.req_id,
            ));
            self.success_latency_stat
                .entry(req.req_type)
                .or_insert_with(|| Histogram::<Time>::new_with_bounds(1, 60 * SECOND, 3).unwrap())
                .record(now() - start_time)
                .unwrap();
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
    read_staleness: Option<Time>,
    num_region: u64,
    retry_count: u64,

    // configs
    read_size_fn: Rc<dyn Fn() -> Time>,
    prewrite_size_fn: Rc<dyn Fn() -> Time>,
    commit_size_fn: Rc<dyn Fn() -> Time>,
    num_queries_fn: Rc<dyn Fn() -> u64>,
    read_only_ratio: f64,
    retry: bool,

    // metrics
    txn_duration_stat: Histogram<Time>,
    failed_txn_stat: HashMap<Error, Histogram<Time>>,
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

#[derive(Default)]
struct TransactionTrace {
    messages: Vec<String>,
}

impl TransactionTrace {
    fn dump(&self) {
        for msg in &self.messages {
            println!("{}", msg);
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
            ));
            commit_req = Some(Request::new(
                start_ts,
                None,
                EventType::CommitRequest,
                commit_size_fn(),
                u64::MAX,
                write_region,
            ));
        }

        let trace = Rc::new(RefCell::new(TransactionTrace::default()));
        trace.borrow_mut().messages.push(format!(
            "{}: txn created with start_ts {}",
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
            rng: StdRng::from_seed(OsRng.gen()),
            rate_exp_dist: Exp::new(cfg.app_config.txn_rate).unwrap(),
            pending_transactions: HashMap::new(),
            txn_duration_stat: Histogram::<Time>::new_with_bounds(NANOSECOND, 60 * SECOND, 3)
                .unwrap(),
            failed_txn_stat: HashMap::new(),
            read_staleness: cfg.app_config.read_staleness,
            num_region: cfg.num_region,
            retry_count: 0,
            read_size_fn: cfg.app_config.read_size_fn.clone(),
            prewrite_size_fn: cfg.app_config.prewrite_size_fn.clone(),
            commit_size_fn: cfg.app_config.commit_size_fn.clone(),
            num_queries_fn: cfg.app_config.num_queries_fn.clone(),
            read_only_ratio: cfg.app_config.read_only_ratio,
            retry: cfg.app_config.retry,
        }
    }

    fn gen_txn(&mut self) {
        // x% read-only, (1-x)% read-only transactions. independent of stale read
        let read_only = rand::random::<f64>() < self.read_only_ratio;
        let mut txn = Transaction::new(
            (self.num_queries_fn)(),
            read_only,
            if read_only { self.read_staleness } else { None },
            self.num_region,
            self.read_size_fn.clone(),
            self.prewrite_size_fn.clone(),
            self.commit_size_fn.clone(),
        );
        let zone = txn.zone;
        let req = txn.remaining_queries.pop_front().unwrap();
        let trace = txn.trace.clone();
        self.pending_transactions.insert(txn.start_ts, txn);
        self.issue_request(zone, req, trace);

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
        txn.trace
            .borrow_mut()
            .messages
            .extend(req.trace.messages.drain(..));

        if let Some(error) = error {
            if self.retry {
                // application retry immediately
                txn.trace.borrow_mut().messages.push(format!(
                    "retrying from app side, now {}, error: {}",
                    now().pretty_print(),
                    error
                ));
                self.retry_count += 1;
                let zone = txn.zone;
                let trace = txn.trace.clone();
                trace
                    .borrow_mut()
                    .messages
                    .extend(req.trace.messages.drain(..));
                self.issue_request(zone, req, trace);
            } else {
                // application doesn't retry
                let txn = self.pending_transactions.remove(&req.start_ts).unwrap();
                txn.trace.borrow_mut().messages.push(format!(
                    "txn failed, now {}, error: {}",
                    now().pretty_print(),
                    error
                ));
                self.failed_txn_stat
                    .entry(error)
                    .or_insert_with(|| {
                        Histogram::<Time>::new_with_bounds(1, 60 * SECOND, 3).unwrap()
                    })
                    .record(now() - txn.start_ts)
                    .unwrap();
            }
        } else {
            // success
            txn.trace.borrow_mut().messages.push(format!(
                "{}: received ok response for req {}",
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
                        self.txn_duration_stat.record(now() - txn.start_ts).unwrap();
                        self.pending_transactions.remove(&req.start_ts);
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
                    self.txn_duration_stat.record(now() - txn.start_ts).unwrap();
                    self.pending_transactions.remove(&req.start_ts);
                }
                CommitPhase::Committed => {
                    unreachable!();
                }
            }
        }
    }

    fn issue_request(&mut self, zone: Zone, req: Request, trace: Rc<RefCell<TransactionTrace>>) {
        trace.borrow_mut().messages.push(format!(
            "{}: sending req {}-{} to zone {}",
            now().pretty_print(),
            req.req_type,
            req.req_id,
            zone,
        ));
        self.events.borrow_mut().push(Event::new(
            now() + rpc_latency(false),
            req.req_type,
            Box::new(move |model: &mut Model| {
                model.find_client_by_zone(zone).on_req(req);
            }),
        ));
    }
}

#[derive(Clone, Default)]
struct RequestTrace {
    messages: Vec<String>,
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
    ) -> Self {
        let req_id = TASK_COUNTER.fetch_add(1, atomic::Ordering::SeqCst);
        Self {
            start_ts,
            stale_read_ts,
            req_type,
            req_id: req_id,
            client_id,
            size,
            region_id: region,
            selector_state: PeerSelectorState::Unknown,
            trace: RequestTrace {
                messages: vec![format!(
                    "{}: req {}-{} created",
                    now().pretty_print(),
                    req_type,
                    req_id,
                )],
            },
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
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ReadTimeout => write!(f, "ReadTimeout"),
            Error::RegionUnavailable => write!(f, "RegionUnavailable"),
            Error::DataIsNotReady => write!(f, "DataIsNotReady"),
        }
    }
}
