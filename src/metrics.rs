use crate::{now, Time, SECOND};
use hdrhistogram::Histogram;
use std::collections::HashMap;
use std::sync::Mutex;

type Label = Vec<String>;
type Name = String;

// Define the TimeBucket type
pub type TimeBucket = Time; // Represents the start of each second
const TIME_BUCKET_RANGE: Time = SECOND;

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

impl MetricGroup {
    fn histogram(&mut self) -> &mut HashMap<Label, HistogramSeries> {
        match self {
            MetricGroup::Histogram(h) => h,
            _ => panic!("wrong type"),
        }
    }

    fn counter(&mut self) -> &mut HashMap<Label, CounterSeries> {
        match self {
            MetricGroup::Counter(c) => c,
            _ => panic!("wrong type"),
        }
    }

    fn gauge(&mut self) -> &mut HashMap<Label, GaugeSeries> {
        match self {
            MetricGroup::Gauge(g) => g,
            _ => panic!("wrong type"),
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
    fn data_point_series(&self) -> Vec<(String, Vec<(f64, f64)>)>;
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

    fn data_point_series(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        let mut res = vec![];
        // mean
        // TODO: introduce an opts to allow more percentiles
        res.push((
            "-mean".to_owned(),
            (0..self.max_time())
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
        res.push((
            "-p99".to_owned(),
            (0..self.max_time())
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

    fn data_point_series(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        vec![(
            String::new(),
            (0..self.max_time())
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

    fn data_point_series(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        match self {
            MetricGroup::Histogram(h) => h
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series()
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
            MetricGroup::Counter(c) => c
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series()
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
            MetricGroup::Gauge(g) => g
                .iter()
                .flat_map(|(label, series)| {
                    series
                        .data_point_series()
                        .into_iter()
                        .map(|(name, points)| (format!("{}{}", label.join("-"), name), points))
                })
                .collect(),
        }
    }
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
        let series = group
            .histogram()
            .entry(label)
            .or_default();
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
        let series = group
            .counter()
            .entry(label)
            .or_default();
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

    pub fn get_hist_group(&self, name: &str) -> Option<MetricGroup> {
        let data = self.data.lock().unwrap();
        data.metric_group.get(name).cloned()
    }

    // sum over all labels of a metric, return the series that is equal to the merge of all series
    pub fn sum_hist_series_by_name(&self, _name: &str) -> HistogramSeries {
        todo!();
        // let mut data = self.data.lock().unwrap();
        // let mut series = HashMap::new();
        // for (key, value) in data.0.iter_mut() {
        //     if key.name == name {
        //         for (time_bucket, histogram) in value.iter_mut() {
        //             let sum_histogram = series.entry(*time_bucket).or_insert_with(|| {
        //                 Histogram::<u64>::new(3).expect("failed to create histogram")
        //             });
        //             sum_histogram
        //                 .add(histogram)
        //                 .expect("failed to add histogram");
        //         }
        //     }
        // }
        // series
    }
}
