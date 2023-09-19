use crate::{now, Time, SECOND};
use hdrhistogram::Histogram;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;

pub type Label = Vec<String>;
pub type Name = String;

// Define the TimeBucket type
pub type TimeBucket = Time; // Represents the start of each second
pub const TIME_BUCKET_RANGE: Time = SECOND;

pub type HistogramSeries = HashMap<TimeBucket, Histogram<Time>>;
pub type CounterSeries = HashMap<TimeBucket, i64>;
pub type GaugeSeries = HashMap<TimeBucket, i64>;

// metrics of the same name and type, possibly different labels.
#[derive(Clone)]
pub enum MetricGroup {
    Histogram(HashMap<Label, HistogramSeries>),
    Counter(HashMap<Label, CounterSeries>),
    Gauge(HashMap<Label, GaugeSeries>),
}

// empty, to draw an emtpy chart
impl Default for MetricGroup {
    fn default() -> Self {
        MetricGroup::Gauge(HashMap::new())
    }
}

impl MetricGroup {
    pub fn histogram(&mut self) -> &mut HashMap<Label, HistogramSeries> {
        match self {
            MetricGroup::Histogram(h) => h,
            _ => panic!("wrong type"),
        }
    }

    pub fn counter(&mut self) -> &mut HashMap<Label, CounterSeries> {
        match self {
            MetricGroup::Counter(c) => c,
            _ => panic!("wrong type"),
        }
    }

    pub fn gauge(&mut self) -> &mut HashMap<Label, GaugeSeries> {
        match self {
            MetricGroup::Gauge(g) => g,
            _ => panic!("wrong type"),
        }
    }

    pub fn max_count(&self) -> u64 {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .map(|(_, series)| series.values().map(|x| x.len()).max().unwrap_or(0))
                .max()
                .unwrap_or(0),
            // these should be unreachable, unless in somewhere like a default empty metric
            MetricGroup::Counter(c) => {
                c.iter().map(|(_, series)| series.len()).max().unwrap_or(0) as u64
            }
            MetricGroup::Gauge(g) => {
                g.iter().map(|(_, series)| series.len()).max().unwrap_or(0) as u64
            }
        }
    }

    pub fn filter_by_label(&self, position: usize, value: &str) -> Self {
        match self {
            MetricGroup::Histogram(h) => {
                let mut res = HashMap::new();
                for (label, series) in h.iter() {
                    if label[position] == value {
                        res.insert(label.clone(), series.clone());
                    }
                }
                MetricGroup::Histogram(res)
            }
            MetricGroup::Counter(c) => {
                let mut res = HashMap::new();
                for (label, series) in c.iter() {
                    if label[position] == value {
                        res.insert(label.clone(), series.clone());
                    }
                }
                MetricGroup::Counter(res)
            }
            MetricGroup::Gauge(g) => {
                let mut res = HashMap::new();
                for (label, series) in g.iter() {
                    if label[position] == value {
                        res.insert(label.clone(), series.clone());
                    }
                }
                MetricGroup::Gauge(res)
            }
        }
    }

    // group by some label, then sum each group.
    // For example, if we group by the zone label, the result could be 3 metrics, the label of each being the AZ name.
    // The other parts of the original labels are lost when aggregating
    pub fn group_by_label(&self, position: usize, aggregate_method: AggregateMethod) -> Self {
        match self {
            MetricGroup::Histogram(h) => {
                let mut res: HashMap<Label, HistogramSeries> = HashMap::new();
                for (label, series) in h.iter() {
                    let new_label = vec![format!("{}-{}", label[position], aggregate_method)];
                    let new_series = res.entry(new_label).or_insert_with(|| series.clone());
                    for (time_bucket, histogram) in series.iter() {
                        let sum_histogram = new_series.entry(*time_bucket).or_insert_with(|| {
                            Histogram::<u64>::new(3).expect("failed to create histogram")
                        });
                        sum_histogram
                            .add(histogram)
                            .expect("failed to add histogram");
                    }
                }
                MetricGroup::Histogram(res)
            }
            MetricGroup::Counter(c) => {
                let mut res = HashMap::new();
                for (label, series) in c.iter() {
                    let new_label = vec![format!("{}-{}", label[position], aggregate_method)];
                    let new_series = res.entry(new_label).or_insert_with(|| series.clone());
                    for (time_bucket, counter) in series.iter() {
                        let sum_counter = new_series.entry(*time_bucket).or_insert(0);
                        match aggregate_method {
                            AggregateMethod::Sum => *sum_counter += counter,
                            AggregateMethod::Mean => unimplemented!(),
                            AggregateMethod::Max => *sum_counter = (*sum_counter).max(*counter),
                            AggregateMethod::Min => *sum_counter = (*sum_counter).min(*counter),
                        }
                    }
                }
                MetricGroup::Counter(res)
            }
            MetricGroup::Gauge(g) => {
                let mut res = HashMap::new();
                for (label, series) in g.iter() {
                    let new_label = vec![format!("{}-{}", label[position], aggregate_method)];
                    let new_series = res.entry(new_label).or_insert_with(|| series.clone());
                    for (time_bucket, gauge) in series.iter() {
                        let sum_gauge = new_series.entry(*time_bucket).or_insert(0);
                        match aggregate_method {
                            AggregateMethod::Sum => *sum_gauge += gauge,
                            AggregateMethod::Mean => unimplemented!(),
                            AggregateMethod::Max => *sum_gauge = (*sum_gauge).max(*gauge),
                            AggregateMethod::Min => *sum_gauge = (*sum_gauge).min(*gauge),
                        }
                    }
                }
                MetricGroup::Gauge(res)
            }
        }
    }
}

// Only for Counter and Gauge. For histogram, we always sum them up
pub enum AggregateMethod {
    Sum,
    Mean,
    Max,
    Min,
}

impl Display for AggregateMethod {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregateMethod::Sum => write!(f, "sum"),
            AggregateMethod::Mean => write!(f, "mean"),
            AggregateMethod::Max => write!(f, "max"),
            AggregateMethod::Min => write!(f, "min"),
        }
    }
}

#[derive(Default)]
pub struct MetricsData {
    metric_group: HashMap<Name, MetricGroup>,
}

pub trait MetricsDataTrait {
    fn max_time(&self) -> usize;
    fn max_value(&self) -> i64;
    fn min_value(&self) -> i64;
    fn data_point_series(&self, max_time: usize, opts: Opts) -> Vec<(String, Vec<(f64, f64)>)>;
}

impl MetricsDataTrait for HistogramSeries {
    fn max_time(&self) -> usize {
        self.len()
    }

    fn max_value(&self) -> i64 {
        self.iter()
            .map(|(_, histogram)| histogram.max() as i64)
            .max()
            .unwrap_or(0)
    }

    fn min_value(&self) -> i64 {
        0
    }

    fn data_point_series(&self, max_time: usize, opts: Opts) -> Vec<(String, Vec<(f64, f64)>)> {
        let mut res = vec![];
        match opts {
            Opts::Hist(HistOpts::Count) => {
                res.push((
                    "-count".to_owned(),
                    (0..max_time)
                        .map(|t| {
                            (
                                t as f64,
                                self.get(&(t as TimeBucket))
                                    .map(|hist| hist.len() as f64)
                                    .unwrap_or(0.0),
                            )
                        })
                        .collect(),
                ));
            }
            Opts::Hist(HistOpts::Percentiles {
                mean,
                p99,
                p999,
                p9999,
            }) => {
                if mean {
                    res.push((
                        "-mean".to_owned(),
                        (0..max_time)
                            .map(|t| {
                                (
                                    t as f64,
                                    self.get(&(t as TimeBucket))
                                        .map(|hist| hist.mean())
                                        .unwrap_or(0.0),
                                )
                            })
                            .collect(),
                    ));
                }
                if p99 {
                    res.push((
                        "-p99".to_owned(),
                        (0..max_time)
                            .map(|t| {
                                (
                                    t as f64,
                                    self.get(&(t as TimeBucket))
                                        .map(|hist| hist.value_at_quantile(0.99) as f64)
                                        .unwrap_or(0.0),
                                )
                            })
                            .collect(),
                    ));
                }
                if p999 {
                    res.push((
                        "-p999".to_owned(),
                        (0..max_time)
                            .map(|t| {
                                (
                                    t as f64,
                                    self.get(&(t as TimeBucket))
                                        .map(|hist| hist.value_at_quantile(0.999) as f64)
                                        .unwrap_or(0.0),
                                )
                            })
                            .collect(),
                    ));
                }
                if p9999 {
                    res.push((
                        "-p9999".to_owned(),
                        (0..max_time)
                            .map(|t| {
                                (
                                    t as f64,
                                    self.get(&(t as TimeBucket))
                                        .map(|hist| hist.value_at_quantile(0.9999) as f64)
                                        .unwrap_or(0.0),
                                )
                            })
                            .collect(),
                    ));
                }
            }
            _ => unreachable!(),
        }
        res
    }
}

impl MetricsDataTrait for CounterSeries {
    fn max_time(&self) -> usize {
        self.len()
    }

    fn max_value(&self) -> i64 {
        self.values().max().copied().unwrap_or(0)
    }

    fn min_value(&self) -> i64 {
        self.values().min().copied().unwrap_or(0).min(0)
    }

    fn data_point_series(&self, max_time: usize, _: Opts) -> Vec<(String, Vec<(f64, f64)>)> {
        vec![(
            String::new(),
            (0..max_time)
                .map(|t| {
                    (
                        t as f64,
                        self.get(&(t as TimeBucket))
                            .map(|x| *x as f64)
                            .unwrap_or(0.0),
                    )
                })
                .collect(),
        )]
    }
}

impl MetricsDataTrait for MetricGroup {
    fn max_time(&self) -> usize {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .map(|(_, series)| series.max_time())
                .max()
                .unwrap_or(0),
            MetricGroup::Counter(c) => c
                .iter()
                .map(|(_, series)| series.max_time())
                .max()
                .unwrap_or(0),
            MetricGroup::Gauge(g) => g
                .iter()
                .map(|(_, series)| series.max_time())
                .max()
                .unwrap_or(0),
        }
    }

    fn max_value(&self) -> i64 {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .map(|(_, series)| series.max_value())
                .max()
                .unwrap_or(0),
            MetricGroup::Counter(c) => c
                .iter()
                .map(|(_, series)| series.max_value())
                .max()
                .unwrap_or(0),
            MetricGroup::Gauge(g) => g
                .iter()
                .map(|(_, series)| series.max_value())
                .max()
                .unwrap_or(0),
        }
    }

    fn min_value(&self) -> i64 {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .map(|(_, series)| series.min_value())
                .min()
                .unwrap_or(0),
            MetricGroup::Counter(c) => c
                .iter()
                .map(|(_, series)| series.min_value())
                .min()
                .unwrap_or(0),
            MetricGroup::Gauge(g) => g
                .iter()
                .map(|(_, series)| series.min_value())
                .min()
                .unwrap_or(0),
        }
    }

    fn data_point_series(&self, max_time: usize, opts: Opts) -> Vec<(String, Vec<(f64, f64)>)> {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series(max_time, opts)
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
            MetricGroup::Counter(c) => c
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series(max_time, opts)
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
            MetricGroup::Gauge(g) => g
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series(max_time, opts)
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
        }
    }
}

#[derive(Copy, Clone)]
pub enum HistOpts {
    Count,
    Percentiles {
        mean: bool,
        p99: bool,
        p999: bool,
        p9999: bool,
    },
}

#[derive(Copy, Clone)]
pub enum Opts {
    Hist(HistOpts),
    Gauge,
    Counter,
}

// Define the MetricsRecorder struct
pub struct MetricsRecorder {
    data: Mutex<MetricsData>,
}

impl Default for MetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRecorder {
    // Constructor
    pub fn new() -> Self {
        MetricsRecorder {
            data: Mutex::new(MetricsData::default()),
        }
    }

    // Record a metric with labels
    pub fn record_hist(&self, name: &str, label: Label, value: u64) {
        let bucket = now() / TIME_BUCKET_RANGE;
        let mut data = self.data.lock().unwrap();
        let group = data
            .metric_group
            .entry(name.to_owned())
            .or_insert_with(|| MetricGroup::Histogram(HashMap::new()));
        let series = group.histogram().entry(label).or_default();
        let histogram = series
            .entry(bucket)
            .or_insert_with(|| Histogram::<u64>::new(3).expect("failed to create histogram"));
        histogram
            .record(value)
            .expect("failed to record value in histogram");
    }

    pub fn inc_counter(&self, name: &str, label: Label, delta: i64) {
        let bucket = now() / TIME_BUCKET_RANGE;
        let mut data = self.data.lock().unwrap();
        let group = data
            .metric_group
            .entry(name.to_owned())
            .or_insert_with(|| MetricGroup::Counter(HashMap::new()));
        let series = group.counter().entry(label).or_default();
        let counter = series.entry(bucket).or_insert(0);
        *counter += delta;
    }

    pub fn set_gauge(&self, name: &str, label: Label, value: i64) {
        let bucket = now() / TIME_BUCKET_RANGE;
        let mut data = self.data.lock().unwrap();
        let group = data
            .metric_group
            .entry(name.to_owned())
            .or_insert_with(|| MetricGroup::Gauge(HashMap::new()));
        let series = group.gauge().entry(label).or_default();
        let gauge = series.entry(bucket).or_insert(0);
        *gauge = value;
    }

    pub fn get_metric_group(&self, name: &str) -> Option<MetricGroup> {
        let data = self.data.lock().unwrap();
        data.metric_group.get(name).cloned()
    }
}
