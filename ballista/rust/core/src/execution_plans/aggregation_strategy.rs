#[derive(Debug, Copy, Clone)]
pub enum AggregationStrategy {
    HashAggregation,
    Coalesce,
}
