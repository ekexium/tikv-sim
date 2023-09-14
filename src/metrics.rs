use crate::{now, Time, TimeTrait, SECOND};
use hdrhistogram::Histogram;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Mutex;

// Define the Label type
type Label = Vec<String>;

// Define the MetricKey struct
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct MetricKey {
    pub name: String,
    pub labels: Label,
}

impl Display for MetricKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.name, self.labels.join(","))
    }
}

// Define the TimeBucket type
pub type TimeBucket = Time; // Represents the start of each second
const TIME_BUCKET_RANGE: Time = SECOND;

// Define the TimeSeries and MetricsData types
pub type TimeSeries = HashMap<TimeBucket, Histogram<Time>>;

pub fn series_max_value(series: &TimeSeries) -> u64 {
    series
        .iter()
        .map(|(_, histogram)| histogram.max())
        .max()
        .unwrap_or(0)
}

pub struct MetricsData(pub HashMap<MetricKey, TimeSeries>);

impl MetricsData {
    pub fn max_time(&self) -> usize {
        self.0
            .iter()
            .map(|(_, series)| series.len())
            .max()
            .unwrap_or(0)
    }

    pub fn max_value(&self) -> u64 {
        self.0
            .iter()
            .map(|(_, series)| {
                series
                    .iter()
                    .map(|(_, histogram)| histogram.max())
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }
}

// Define the MetricsRecorder struct
pub struct MetricsRecorder {
    data: Mutex<MetricsData>,
}

impl MetricsRecorder {
    // Constructor
    pub fn new() -> Self {
        MetricsRecorder {
            data: Mutex::new(MetricsData(HashMap::new())),
        }
    }

    // Record a metric with labels
    pub fn record(&self, metric_name: &str, labels: Label, value: u64) {
        let bucket = now() / TIME_BUCKET_RANGE;
        let key = MetricKey {
            name: metric_name.to_string(),
            labels,
        };
        let mut data = self.data.lock().unwrap();
        let series = data.0.entry(key).or_insert_with(TimeSeries::new);
        let histogram = series
            .entry(bucket)
            .or_insert_with(|| Histogram::<u64>::new(3).expect("failed to create histogram"));
        histogram
            .record(value)
            .expect("failed to record value in histogram");
    }

    // Retrieve the P99 value for a given metric, set of labels, and time bucket
    pub fn p99_for_bucket(
        &self,
        metric_name: &str,
        labels: &Label,
        bucket: TimeBucket,
    ) -> Option<u64> {
        let key = MetricKey {
            name: metric_name.to_string(),
            labels: labels.clone(),
        };
        if let Some(series) = self.data.lock().unwrap().0.get(&key) {
            if let Some(histogram) = series.get(&bucket) {
                return Some(histogram.value_at_quantile(0.99));
            } else {
                return None;
            }
        } else {
            panic!("no data found for metric {}-{:?}", metric_name, labels)
        }
    }

    pub fn get_by_name(&self, name: &str) -> MetricsData {
        MetricsData(
            self.data
                .lock()
                .unwrap()
                .0
                .iter()
                .filter(|(key, _)| key.name == name)
                .map(|(key, series)| (key.clone(), series.clone()))
                .collect(),
        )
    }

    // sum over all labels of a metric, return the series that is equal to the merge of all series
    pub fn get_sum_by_name(&self, name: &str) -> TimeSeries {
        let mut data = self.data.lock().unwrap();
        let mut series = HashMap::new();
        for (key, value) in data.0.iter_mut() {
            if key.name == name {
                for (time_bucket, histogram) in value.iter_mut() {
                    let sum_histogram = series.entry(*time_bucket).or_insert_with(|| {
                        Histogram::<u64>::new(3).expect("failed to create histogram")
                    });
                    sum_histogram
                        .add(histogram)
                        .expect("failed to add histogram");
                }
            }
        }
        series
    }

    // Filter metrics data by a specific label and its value
    pub fn filter_by_label(&self, name: &str, label: Vec<String>) -> MetricsData {
        MetricsData(
            self.data
                .lock()
                .unwrap()
                .0
                .iter()
                .filter(|(key, _)| key.name == name && key.labels == label)
                .map(|(key, series)| (key.clone(), series.clone()))
                .collect(),
        )
    }
}
