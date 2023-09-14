use hdrhistogram::Histogram;
use indicatif::ProgressBar;
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::cmp::{max, Ordering};
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
use tikv_sim::TimeTrait;
use tikv_sim::{Time, *};

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

lazy_static! {
    static ref METRICS: MetricsRecorder = MetricsRecorder::new();
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
        num_servers: 3,
        max_time: 500 * SECOND,
        metrics_interval: SECOND,
        enable_trace: false,
        server_config: ServerConfig {
            enable_async_commit: true,
            num_read_workers: 1,
            num_write_workers: 10,
            read_timeout: SECOND,
            advance_interval: 5 * SECOND,
            broadcast_interval: 5 * SECOND,
        },
        client_config: ClientConfig {
            max_execution_time: 10 * SECOND,
        },
        app_config: AppConfig {
            retry: false,
            txn_rate: 300.0,
            read_staleness: Some(12 * SECOND),
            read_size_fn: Rc::new(|| (rand::random::<u64>() % 5 + 1) * MILLISECOND),
            prewrite_size_fn: Rc::new(|| (rand::random::<u64>() % 30 + 1) * MILLISECOND),
            commit_size_fn: Rc::new(|| (rand::random::<u64>() % 20 + 1) * MILLISECOND),
            num_queries_fn: Rc::new(|| 5),
            read_only_ratio: 1.0,
        },
    };
    assert_eq!(config.metrics_interval, SECOND, "why bother");
    assert_eq!(config.num_servers, 3, "not supported yet");

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
            Client::new(Zone::AZ1, events.clone(), &config),
            Client::new(Zone::AZ2, events.clone(), &config),
            Client::new(Zone::AZ3, events.clone(), &config),
        ],
        app: App::new(events.clone(), &config),
        app_ok_transaction_durations: vec![],
        app_fail_transaction_durations: vec![],
        kv_ok_durations: Default::default(),
        kv_error_durations: Default::default(),
        server_max_resolved_ts_gap: vec![],
        server_read_queue_length: Default::default(),
        server_write_queue_length: Default::default(),
        advance_resolved_ts_failure_for_lock_cf: vec![],
        advance_resolved_ts_failure_for_memory_lock: vec![],
        server_read_worker_busy_time: vec![],
        server_write_worker_busy_time: vec![],
        server_read_req_count: vec![],
        server_write_req_count: vec![],
        server_error_count: vec![],
    };
    model.init(&config);
    model.inject_io_delay();

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
    draw_metrics(&model, &config)?;
    Ok(())
}

fn draw_metrics(model: &Model, cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let num_graphs = 16usize + 2 * 3/* num_server*/;
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
        let metrics = METRICS
            .get_hist_group("app_ok_txn_duration")
            .expect("no app ok records");

        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("successful txn latency", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..metrics.max_time() as f32,
                0f32..(metrics.max_value() as f32 / y_unit as f32) * 1.2,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;

        // mean latency
        for (i, (name, points)) in metrics.data_point_series().iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    points
                        .iter()
                        .map(|(x, y)| (*x as f32, *y as f32 / y_unit as f32)),
                    colors[i],
                ))?
                .label(name)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        // for (i, (key, series)) in metrics.0.iter().enumerate() {
        //     chart
        //         .draw_series(LineSeries::new(
        //             (0..series.len()).map(|t| {
        //                 (
        //                     t as f32,
        //                     series
        //                         .get(&(t as Time))
        //                         .map(|hist| hist.mean() as f32 / y_unit as f32)
        //                         .unwrap_or(0.0),
        //                 )
        //             }),
        //             colors[i],
        //         ))?
        //         .label(key.to_string() + "-mean")
        //         .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        // }

        // p99 latency
        // for (i, (key, series)) in metrics.0.iter().enumerate() {
        //     chart
        //         .draw_series(PointSeries::<(f32, f32), _, Circle<_, _>, f32>::new(
        //             (0..series.len()).map(|t| {
        //                 (
        //                     t as f32,
        //                     series
        //                         .get(&(t as Time))
        //                         .map(|hist| hist.value_at_quantile(0.99) as f32 / y_unit as f32)
        //                         .unwrap_or(0.0),
        //                 )
        //             }),
        //             3f32,
        //             colors[i],
        //         ))?
        //         .label(key.to_string() + "-p99")
        //         .legend(move |(x, y)| Circle::new((x, y), 3f32, colors[i]));
        // }

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
                0f32..model.advance_resolved_ts_failure_for_lock_cf.len() as f32,
                0f32..(*model
                    .advance_resolved_ts_failure_for_lock_cf
                    .iter()
                    .map(|x| x.values().max().unwrap_or(&0))
                    .max()
                    .unwrap_or(&0) as f32
                    * 1.2)
                    .max(1.0)
                    .max(
                        *model
                            .advance_resolved_ts_failure_for_memory_lock
                            .iter()
                            .map(|x| x.values().max().unwrap_or(&0))
                            .max()
                            .unwrap_or(&0) as f32
                            * 1.2,
                    ),
            )?;

        chart.configure_mesh().disable_mesh().draw()?;
        let server_ids = model
            .advance_resolved_ts_failure_for_lock_cf
            .iter()
            .flat_map(|x| x.keys())
            .collect::<HashSet<_>>();

        for (i, server_id) in server_ids.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    xs.clone().zip(
                        model
                            .advance_resolved_ts_failure_for_lock_cf
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32),
                    ),
                    colors[i],
                ))?
                .label(format!("server-{}-lock", server_id))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));

            chart
                .draw_series(PointSeries::<_, _, Circle<_, _>, _>::new(
                    xs.clone().zip(
                        model
                            .advance_resolved_ts_failure_for_memory_lock
                            .iter()
                            .map(|x| x.get(server_id).copied().unwrap_or(0) as f32),
                    ),
                    3,
                    colors[i],
                ))?
                .label(format!("server-{}-memory-lock", server_id))
                .legend(move |(x, y)| Circle::new((x, y), 3, colors[i]));
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

    // server error rates
    {
        for server in &model.servers {
            chart_id += 1;
            let mut chart = ChartBuilder::on(&children_area[chart_id])
                .caption(
                    format!("server-{} error rates (per interval)", server.server_id),
                    font.clone(),
                )
                .margin(30)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(
                    0f32..model.server_error_count.len() as f32,
                    0f32..(*model
                        .server_error_count
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
            let errors = model
                .server_error_count
                .iter()
                .flat_map(|x| {
                    x.get(&server.server_id)
                        .map(|x| x.keys().copied().collect::<HashSet<_>>())
                        .unwrap_or_default()
                })
                .collect::<HashSet<_>>();

            for (i, error) in errors.iter().enumerate() {
                chart
                    .draw_series(LineSeries::new(
                        xs.clone().zip(model.server_error_count.iter().map(|x| {
                            x.get(&server.server_id)
                                .map(|x| x.get(error).copied().unwrap_or(0) as f32)
                                .unwrap_or(0.0)
                        })),
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
    }

    // read schedule wait stat
    {
        chart_id += 1;
        let metrics = METRICS
            .get_hist_group("server_read_schedule_wait")
            .expect("no stats in server read schedule wait");
        let (y_label, y_unit) = ("ms", MILLISECOND);

        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("read schedule wait stat", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .build_cartesian_2d(
                0f32..metrics.max_time() as f32,
                0f32..metrics.max_value() as f32 * 1.2 / y_unit as f32,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;

        // mean latency
        for (i, (name, points)) in metrics.data_point_series().iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    points
                        .iter()
                        .map(|(x, y)| (*x as f32, *y as f32 / y_unit as f32)),
                    colors[i],
                ))?
                .label(name.to_string())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
        }

        // TODO: p99 latency

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.5))
            .draw()?;
    }

    // server_rest_time_to_handle_read
    {
        let metrics = METRICS
            .get_hist_group("server_rest_time_to_handle_read")
            .expect("no stats in server rest time to handle read");
        let (y_label, y_unit) = ("ms", MILLISECOND);

        chart_id += 1;
        let mut chart = ChartBuilder::on(&children_area[chart_id])
            .caption("server rest time to handle read", font.clone())
            .margin(30)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .build_cartesian_2d(
                0f32..metrics.max_time() as f32,
                0f32..metrics.max_value() as f32 * 1.2 / y_unit as f32,
            )?;

        chart
            .configure_mesh()
            .disable_mesh()
            .y_label_formatter(&|x| format!("{:.1}{}", x, y_label))
            .draw()?;

        for (i, (name, points)) in metrics.data_point_series().iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    points
                        .iter()
                        .map(|(x, y)| (*x as f32, *y as f32 / y_unit as f32)),
                    colors[i],
                ))?
                .label(name.to_string())
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
    advance_resolved_ts_failure_for_lock_cf: Vec<HashMap<u64, u64>>,
    advance_resolved_ts_failure_for_memory_lock: Vec<HashMap<u64, u64>>,
    // server id -> total time
    server_read_worker_busy_time: Vec<HashMap<u64, u64>>,
    // server id -> total time
    server_write_worker_busy_time: Vec<HashMap<u64, u64>>,
    server_read_req_count: Vec<HashMap<u64, HashMap<PeerSelectorState, u64>>>,
    server_write_req_count: Vec<HashMap<u64, HashMap<PeerSelectorState, u64>>>,
    server_error_count: Vec<HashMap<u64, HashMap<Error, u64>>>,
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
                let client = Model::find_client_by_zone(model, Zone::AZ1);
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
                let client = Model::find_client_by_zone(model, Zone::AZ1);
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

    fn inject_io_delay(&mut self) {
        self.events.borrow_mut().push(Event::new(
            100 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.servers[0].read_delay = SECOND;
            }),
        ));
        self.events.borrow_mut().push(Event::new(
            110 * SECOND,
            EventType::Injection,
            Box::new(move |model| {
                model.servers[0].read_delay = 0;
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
                fail_advance_resolved_ts_lock_cf_count: 0,
                fail_advance_resolved_ts_memory_lock_count: 0,
            };
            leader.update_resolved_ts(self.events.clone(), None);
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
                        fail_advance_resolved_ts_lock_cf_count: 0,
                        fail_advance_resolved_ts_memory_lock_count: 0,
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
                        .or_insert_with(new_hist)
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
                        .or_insert_with(new_hist)
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
                            peer.fail_advance_resolved_ts_lock_cf_count;
                        peer.fail_advance_resolved_ts_lock_cf_count = 0;
                    }
                }
            }
            self.advance_resolved_ts_failure_for_lock_cf.push(map);
        }

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                for peer in &mut server.peers.values_mut() {
                    if peer.role == Role::Leader {
                        *map.entry(server.server_id).or_insert(0) +=
                            peer.fail_advance_resolved_ts_memory_lock_count;
                        peer.fail_advance_resolved_ts_memory_lock_count = 0;
                    }
                }
            }
            self.advance_resolved_ts_failure_for_memory_lock.push(map);
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

        {
            let mut map = HashMap::new();
            for server in &mut self.servers {
                map.insert(server.server_id, server.error_count.drain().collect());
            }
            self.server_error_count.push(map);
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
    fail_advance_resolved_ts_lock_cf_count: u64,
    fail_advance_resolved_ts_memory_lock_count: u64,
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
                self.fail_advance_resolved_ts_lock_cf_count += 1;
            } else if candidate == min_memory_lock.unwrap() {
                self.fail_advance_resolved_ts_memory_lock_count += 1;
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
    error_count: HashMap<Error, u64>,
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
            read_schedule_wait_stat: new_hist(),
            concurrency_manager: Default::default(),
            read_timeout: cfg.server_config.read_timeout,
            enable_async_commit: cfg.server_config.enable_async_commit,
            read_delay: 0,
            write_schedule_wait_stat: new_hist(),
            read_worker_time: 0,
            write_worker_time: 0,
            error_count: Default::default(),
            read_req_count: HashMap::new(),
            write_req_count: HashMap::new(),
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
        METRICS.record_hist(
            "server_read_schedule_wait",
            vec![self.server_id.to_string()],
            now() - accept_time,
        );
        self.read_schedule_wait_stat
            .record(now() - accept_time)
            .unwrap();

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

                *self.error_count.entry(Error::DataIsNotReady).or_default() += 1;

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
        METRICS.record_hist(
            "server_rest_time_to_handle_read",
            vec![this_server_id.to_string()],
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
                    *this.error_count.entry(Error::ReadTimeout).or_default() += 1;

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
            latency_stat: new_hist(),
            error_latency_stat: HashMap::new(),
            success_latency_stat: HashMap::new(),
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
                        this.error_latency_stat
                            .entry(Error::MaxExecutionTimeExceeded)
                            .or_insert_with(new_hist)
                            .record(now() - start_time)
                            .unwrap();
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
        self.latency_stat.record(now() - start_time).unwrap();

        if let Some(e) = error {
            req.trace.record(format!(
                "{}: client {}, received error {} for req {}",
                now().pretty_print(),
                self.id,
                e,
                req.req_id,
            ));
            self.error_latency_stat
                .entry(e)
                .or_insert_with(new_hist)
                .record(now() - start_time)
                .unwrap();
            // retry other peers
            self.issue_request(req, selector.clone());
        } else {
            req.trace.record(format!(
                "{}: client {}, received success for req {}",
                now().pretty_print(),
                self.id,
                req.req_id,
            ));
            self.success_latency_stat
                .entry(req.req_type)
                .or_insert_with(new_hist)
                .record(now() - start_time)
                .unwrap();
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
    stop: bool,

    // configs
    read_size_fn: Rc<dyn Fn() -> Time>,
    prewrite_size_fn: Rc<dyn Fn() -> Time>,
    commit_size_fn: Rc<dyn Fn() -> Time>,
    num_queries_fn: Rc<dyn Fn() -> u64>,
    read_only_ratio: f64,
    retry: bool,
    enable_trace: bool,

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
            failed_txn_stat: HashMap::new(),
            read_staleness: cfg.app_config.read_staleness,
            num_region: cfg.num_region,
            read_size_fn: cfg.app_config.read_size_fn.clone(),
            prewrite_size_fn: cfg.app_config.prewrite_size_fn.clone(),
            commit_size_fn: cfg.app_config.commit_size_fn.clone(),
            num_queries_fn: cfg.app_config.num_queries_fn.clone(),
            read_only_ratio: cfg.app_config.read_only_ratio,
            retry: cfg.app_config.retry,
            enable_trace: cfg.enable_trace,
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
        self.events.borrow_mut().push(Event::new(
            now() + interval,
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
                METRICS.inc_counter("app_retry", vec![zone.to_string()], 1);
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
                self.failed_txn_stat
                    .entry(error)
                    .or_insert_with(new_hist)
                    .record(now() - txn.start_ts)
                    .unwrap();
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
                        METRICS.record_hist(
                            "app_ok_txn_duration",
                            vec![txn.zone.to_string()],
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
                    METRICS.record_hist(
                        "app_ok_txn_duration",
                        vec![txn.zone.to_string()],
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
                model.find_client_by_zone(zone).on_req(req);
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
