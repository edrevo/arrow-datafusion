// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Distributed query execution
//!
//! This code is EXPERIMENTAL and still under development

use std::collections::HashMap;
use std::sync::Arc;

use ballista_core::datasource::DfTableAdapter;
use ballista_core::error::{BallistaError, Result};
use ballista_core::execution_plans::AggregationStrategy;
use ballista_core::{
    execution_plans::{QueryStageExec, ShuffleReaderExec, UnresolvedShuffleExec},
    serde::scheduler::PartitionLocation,
};
use datafusion::execution::context::{ExecutionConfig, ExecutionContext};
use datafusion::physical_optimizer::coalesce_batches::CoalesceBatches;
use datafusion::physical_optimizer::merge_exec::AddMergeExec;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use log::info;

type PartialQueryStageResult = (Arc<dyn ExecutionPlan>, Vec<Arc<QueryStageExec>>);

pub struct DistributedPlanner {
    next_stage_id: usize,
}

impl DistributedPlanner {
    pub fn new() -> Self {
        Self { next_stage_id: 0 }
    }
}

impl Default for DistributedPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedPlanner {
    /// Returns a vector of ExecutionPlans, where the root node is a [QueryStageExec].
    /// Plans that depend on the input of other plans will have leaf nodes of type [UnresolvedShuffleExec].
    /// A [QueryStageExec] is created whenever the partitioning changes.
    ///
    /// Returns an empty vector if the execution_plan doesn't need to be sliced into several stages.
    pub fn plan_query_stages(
        &mut self,
        job_id: &str,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<Arc<QueryStageExec>>> {
        info!("planning query stages");
        let (new_plan, mut stages) =
            self.plan_query_stages_internal(job_id, execution_plan)?;
        stages.push(create_query_stage(job_id, self.next_stage_id(), new_plan)?);
        Ok(stages)
    }

    /// Returns a potentially modified version of the input execution_plan along with the resulting query stages.
    /// This function is needed because the input execution_plan might need to be modified, but it might not hold a
    /// compelte query stage (its parent might also belong to the same stage)
    fn plan_query_stages_internal(
        &mut self,
        job_id: &str,
        execution_plan: Arc<dyn ExecutionPlan>,
    ) -> Result<PartialQueryStageResult> {
        // recurse down and replace children
        if execution_plan.children().is_empty() {
            return Ok((execution_plan, vec![]));
        }

        let mut stages = vec![];
        let mut children = vec![];
        for child in execution_plan.children() {
            let (new_child, mut child_stages) =
                self.plan_query_stages_internal(job_id, child.clone())?;
            children.push(new_child);
            stages.append(&mut child_stages);
        }

        if let Some(adapter) = execution_plan.as_any().downcast_ref::<DfTableAdapter>() {
            // remove Repartition rule because that isn't supported yet
            let rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> = vec![
                Arc::new(CoalesceBatches::new()),
                Arc::new(AddMergeExec::new()),
            ];
            let config = ExecutionConfig::new().with_physical_optimizer_rules(rules);
            let ctx = ExecutionContext::with_config(config);
            Ok((ctx.create_physical_plan(&adapter.logical_plan)?, stages))
        } else if execution_plan.requires_distributed_shuffle() {
            let mut new_children: Vec<Arc<dyn ExecutionPlan>> = vec![];
            let aggregation_strategy = match execution_plan.required_child_distribution()
            {
                datafusion::physical_plan::Distribution::UnspecifiedDistribution => {
                    AggregationStrategy::Coalesce
                }
                datafusion::physical_plan::Distribution::SinglePartition => {
                    AggregationStrategy::Coalesce
                }
                datafusion::physical_plan::Distribution::HashPartitioned(_) => {
                    AggregationStrategy::HashAggregation
                }
            };
            for child in &children {
                let new_stage =
                    create_query_stage(job_id, self.next_stage_id(), child.clone())?;
                new_children.push(Arc::new(UnresolvedShuffleExec::new(
                    vec![new_stage.stage_id()],
                    new_stage.schema().clone(),
                    new_stage.output_partitioning().partition_count(),
                    aggregation_strategy,
                )));
                stages.push(new_stage);
            }
            Ok((execution_plan.with_new_children(new_children)?, stages))
        } else {
            Ok((execution_plan.with_new_children(children)?, stages))
        }
    }

    /// Generate a new stage ID
    fn next_stage_id(&mut self) -> usize {
        self.next_stage_id += 1;
        self.next_stage_id
    }
}

pub fn remove_unresolved_shuffles(
    stage: &dyn ExecutionPlan,
    partition_locations: &HashMap<usize, Vec<Vec<PartitionLocation>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut new_children: Vec<Arc<dyn ExecutionPlan>> = vec![];
    for child in stage.children() {
        if let Some(unresolved_shuffle) =
            child.as_any().downcast_ref::<UnresolvedShuffleExec>()
        {
            let mut relevant_locations = vec![];
            for id in &unresolved_shuffle.query_stage_ids {
                relevant_locations.append(
                    &mut partition_locations
                        .get(id)
                        .ok_or_else(|| {
                            BallistaError::General(
                                "Missing partition location. Could not remove unresolved shuffles"
                                    .to_owned(),
                            )
                        })?
                        .clone(),
                );
            }
            new_children.push(Arc::new(ShuffleReaderExec::try_new(
                relevant_locations,
                unresolved_shuffle.schema().clone(),
            )?))
        } else {
            new_children.push(remove_unresolved_shuffles(
                child.as_ref(),
                partition_locations,
            )?);
        }
    }
    Ok(stage.with_new_children(new_children)?)
}

fn create_query_stage(
    job_id: &str,
    stage_id: usize,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<QueryStageExec>> {
    Ok(Arc::new(QueryStageExec::try_new(
        job_id.to_owned(),
        stage_id,
        plan,
        "".to_owned(), // executor will decide on the work_dir path
    )?))
}

#[cfg(test)]
mod test {
    use crate::planner::DistributedPlanner;
    use crate::test_utils::datafusion_test_context;
    use ballista_core::error::BallistaError;
    use ballista_core::execution_plans::UnresolvedShuffleExec;
    use ballista_core::serde::protobuf;
    use datafusion::physical_plan::hash_aggregate::HashAggregateExec;
    use datafusion::physical_plan::sort::SortExec;
    use datafusion::physical_plan::{displayable, ExecutionPlan};
    use datafusion::physical_plan::{merge::MergeExec, projection::ProjectionExec};
    use std::convert::TryInto;
    use std::sync::Arc;
    use uuid::Uuid;

    macro_rules! downcast_exec {
        ($exec: expr, $ty: ty) => {
            $exec.as_any().downcast_ref::<$ty>().unwrap()
        };
    }

    #[test]
    fn test() -> Result<(), BallistaError> {
        let mut ctx = datafusion_test_context("testdata")?;

        // simplified form of TPC-H query 1
        let df = ctx.sql(
            "select l_returnflag, sum(l_extendedprice * 1) as sum_disc_price
            from lineitem
            group by l_returnflag
            order by l_returnflag",
        )?;

        let plan = df.to_logical_plan();
        let plan = ctx.optimize(&plan)?;
        let plan = ctx.create_physical_plan(&plan)?;

        /*let mut current_plan = plan.clone();
        println!("{:?} / {:?}", plan, plan.output_partitioning());
        while !current_plan.children().is_empty() {
            current_plan = current_plan.children()[0].clone();
            println!("{:?} / {:?}", current_plan, current_plan.output_partitioning());
        }*/

        let mut planner = DistributedPlanner::new();
        let job_uuid = Uuid::new_v4();
        let stages = planner.plan_query_stages(&job_uuid.to_string(), plan)?;
        for stage in &stages {
            println!("{}", displayable(stage.as_ref()).indent().to_string());
        }

        /* Expected result:
        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=1
         HashAggregateExec: groupBy=["l_returnflag"], aggrExpr=["SUM(l_extendedprice Multiply Int64(1)) [\"l_extendedprice * CAST(1 AS Float64)\"]"]
          CsvExec: testdata/lineitem; partitions=2
        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=2
         MergeExec
          UnresolvedShuffleExec: stages=[1]
        QueryStageExec: job=f011432e-e424-4016-915d-e3d8b84f6dbd, stage=3
         SortExec { input: ProjectionExec { expr: [(Column { name: "l_returnflag" }, "l_returnflag"), (Column { name: "SUM(l_ext
          ProjectionExec { expr: [(Column { name: "l_returnflag" }, "l_returnflag"), (Column { name: "SUM(l_extendedprice Multip
           HashAggregateExec: groupBy=["l_returnflag"], aggrExpr=["SUM(l_extendedprice Multiply Int64(1)) [\"l_extendedprice * CAST(1 AS Float64)\"]"]
            UnresolvedShuffleExec: stages=[2]
        */

        let sort = stages[2].children()[0].clone();
        let sort = downcast_exec!(sort, SortExec);

        let projection = sort.children()[0].clone();
        println!("{:?}", projection);
        let projection = downcast_exec!(projection, ProjectionExec);

        let final_hash = projection.children()[0].clone();
        let final_hash = downcast_exec!(final_hash, HashAggregateExec);

        let unresolved_shuffle = final_hash.children()[0].clone();
        let unresolved_shuffle =
            downcast_exec!(unresolved_shuffle, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle.query_stage_ids, vec![2]);

        let merge_exec = stages[1].children()[0].clone();
        let merge_exec = downcast_exec!(merge_exec, MergeExec);

        let unresolved_shuffle = merge_exec.children()[0].clone();
        let unresolved_shuffle =
            downcast_exec!(unresolved_shuffle, UnresolvedShuffleExec);
        assert_eq!(unresolved_shuffle.query_stage_ids, vec![1]);

        let partial_hash = stages[0].children()[0].clone();
        let partial_hash_serde = roundtrip_operator(partial_hash.clone())?;

        let partial_hash = downcast_exec!(partial_hash, HashAggregateExec);
        let partial_hash_serde = downcast_exec!(partial_hash_serde, HashAggregateExec);

        assert_eq!(
            format!("{:?}", partial_hash),
            format!("{:?}", partial_hash_serde)
        );

        Ok(())
    }

    fn roundtrip_operator(
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>, BallistaError> {
        let proto: protobuf::PhysicalPlanNode = plan.clone().try_into()?;
        let result_exec_plan: Arc<dyn ExecutionPlan> = (&proto).try_into()?;
        Ok(result_exec_plan)
    }
}
