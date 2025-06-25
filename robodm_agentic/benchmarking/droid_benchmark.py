#!/usr/bin/env python3
"""
DROID Dataset Benchmark for RoboDM Agentic Framework

This script benchmarks the performance of robodm-agentic with 1000 DROID trajectories.
It measures ingestion time, query performance, and provides detailed metrics.
"""

import os
import sys
import time
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directories to path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent.parent
sys.path.insert(0, str(robodm_root))
sys.path.insert(0, str(current_dir.parent))

try:
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Prevent tensorflow from allocating GPU memory
    tf.config.set_visible_devices([], "GPU")
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Please install: pip install tensorflow tensorflow-datasets numpy")
    sys.exit(1)

import robodm
from robodm import Trajectory

# Import robodm-agentic components
from robodm_agentic.core.robodm_interface import RoboDMInterface
from robodm_agentic.core.agent import RoboDMAgent
from robodm_agentic.clients.llm_client import LLMClient
from robodm_agentic.clients.vlm_client import VLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DROIDBenchmark:
    """Benchmark class for testing RoboDM Agentic with DROID dataset."""
    
    def __init__(self, 
                 num_trajectories: int = 1000,
                 output_dir: Optional[str] = None,
                 max_workers: int = 4):
        """Initialize benchmark.
        
        Args:
            num_trajectories: Number of trajectories to ingest
            output_dir: Directory to save trajectories (default: temp dir)
            max_workers: Number of parallel workers for ingestion
        """
        self.num_trajectories = num_trajectories
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="droid_benchmark_"))
        self.max_workers = max_workers
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            'ingestion_times': [],
            'query_times': [],
            'code_generation_times': [],
            'execution_times': [],
            'vision_analysis_times': [],
            'memory_usage': [],
            'trajectory_sizes': [],
            'total_ingestion_time': 0.0,
            'total_query_time': 0.0
        }
        
        # Benchmark queries
        self.benchmark_queries = [
            # Basic queries
            "How many trajectories do we have?",
            "Find all successful trajectories",
            "Show me 10 random trajectories",
            "Count trajectories by length",
            
            # Visual queries
            "Find trajectories with hidden views",
            "Show me frames where the robot is grasping",
            "Find trajectories with red objects",
            "Analyze robot actions in trajectories",
            
            # Complex queries
            "Compare successful vs failed trajectories",
            "Find trajectories longer than 100 timesteps",
            "Show me the beginning and end of trajectories",
            "Find trajectories with specific features",
            
            # Performance queries
            "Get all trajectory metadata",
            "Sample 50 trajectories and analyze their frames",
            "Find trajectories with error conditions"
        ]
    
    def _transpose_list_of_dicts(self, list_of_dicts):
        """Converts a list of nested dictionaries to a nested dictionary of lists."""
        if not list_of_dicts:
            return {}

        if not isinstance(list_of_dicts[0], dict):
            return list_of_dicts

        dict_of_lists = {}
        for key in list_of_dicts[0].keys():
            dict_of_lists[key] = self._transpose_list_of_dicts(
                [d[key] for d in list_of_dicts]
            )
        return dict_of_lists
    
    def _ingest_single_trajectory(self, episode_data: tuple) -> Optional[Dict[str, Any]]:
        """Ingest a single trajectory (for parallel processing)."""
        i, episode = episode_data
        
        try:
            episode_start_time = time.time()
            
            # Convert episode to list of steps
            steps_list = list(episode["steps"])
            
            if not steps_list:
                logger.warning(f"Episode {i} is empty, skipping...")
                return None
            
            # Transpose to dictionary of lists
            episode_steps = self._transpose_list_of_dicts(steps_list)
            
            # Save as RoboDM trajectory
            trajectory_path = self.output_dir / f"droid_trajectory_{i:04d}.vla"
            
            Trajectory.from_dict_of_lists(
                data=episode_steps, 
                path=str(trajectory_path), 
                video_codec="libx264"
            )
            
            # Get trajectory size
            file_size = trajectory_path.stat().st_size / (1024 * 1024)  # MB
            episode_time = time.time() - episode_start_time
            
            return {
                'trajectory_path': str(trajectory_path),
                'ingestion_time': episode_time,
                'file_size': file_size,
                'index': i
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest episode {i}: {e}")
            return None
    
    async def ingest_droid_trajectories(self) -> List[str]:
        """Ingest DROID trajectories and convert to RoboDM format using parallel processing."""
        logger.info(f"Starting parallel ingestion of {self.num_trajectories} DROID trajectories...")
        
        start_time = time.time()
        trajectory_paths = []
        
        try:
            # Load DROID dataset
            logger.info("Loading DROID dataset from tensorflow_datasets...")
            builder = tfds.builder_from_directory(builder_dir=
                "gs://gresearch/robotics/fractal20220817_data/0.1.0"
            )
            
            # Load episodes from training split
            ds = builder.as_dataset(split=f"train[:{self.num_trajectories}]")
            
            # Convert to list for parallel processing
            episodes = list(tfds.as_numpy(ds))
            logger.info(f"Loaded {len(episodes)} episodes for processing")
            
            # Process episodes in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_episode = {
                    executor.submit(self._ingest_single_trajectory, (i, episode)): i 
                    for i, episode in enumerate(episodes)
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_episode):
                    episode_idx = future_to_episode[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        if result:
                            trajectory_paths.append(result['trajectory_path'])
                            self.metrics['ingestion_times'].append(result['ingestion_time'])
                            self.metrics['trajectory_sizes'].append(result['file_size'])
                        
                        if completed % 100 == 0:
                            logger.info(f"Processed {completed}/{len(episodes)} episodes...")
                            
                    except Exception as e:
                        logger.error(f"Error processing episode {episode_idx}: {e}")
            
            self.metrics['total_ingestion_time'] = time.time() - start_time
            
            logger.info(f"Successfully ingested {len(trajectory_paths)} trajectories")
            logger.info(f"Total ingestion time: {self.metrics['total_ingestion_time']:.2f}s")
            logger.info(f"Average ingestion time per trajectory: {statistics.mean(self.metrics['ingestion_times']):.3f}s")
            logger.info(f"Average trajectory size: {statistics.mean(self.metrics['trajectory_sizes']):.2f}MB")
            logger.info(f"Parallel processing with {self.max_workers} workers")
            
            return trajectory_paths
            
        except Exception as e:
            logger.error(f"Failed to ingest DROID trajectories: {e}")
            return []
    
    async def setup_agent(self, trajectory_paths: List[str]) -> Optional[RoboDMAgent]:
        """Set up RoboDM agent with ingested trajectories using real models."""
        logger.info("Setting up RoboDM agent with real models...")
        
        # Create RoboDM interface
        robodm_interface = RoboDMInterface(str(self.output_dir))
        
        # Initialize clients with real models (following example_usage.py pattern)
        llm_client = None
        vlm_client = None
        
        try:
            llm_client = LLMClient(
                model="qwen2.5:7b",  # Adjust model as needed
                provider="ollama"    # or "openai" with appropriate API key
            )
            logger.info("✓ LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize LLM client: {e}")
            logger.error("Please install ollama or configure OpenAI API key")
            return None

        try:
            vlm_client = VLMClient(
                model="llava:7b",    # Adjust model as needed
                provider="ollama"
            )
            logger.info("✓ VLM client initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize VLM client: {e}")
            vlm_client = None
        
        # Create agent
        agent = RoboDMAgent(
            robodm_interface=robodm_interface,
            llm_client=llm_client,
            vlm_client=vlm_client,
            enable_vision=vlm_client is not None
        )
        
        # Test setup
        test_results = await agent.test_setup()
        logger.info("Agent setup test results:")
        for component, status in test_results.items():
            status_str = "✓" if status else "✗"
            logger.info(f"  {status_str} {component}: {'OK' if status else 'Failed'}")
        
        if not any(test_results.values()):
            logger.error("Setup failed. Please check your configuration.")
            return None
        
        return agent
    
    async def run_benchmark_queries(self, agent: RoboDMAgent) -> Dict[str, Any]:
        """Run benchmark queries and measure performance."""
        logger.info(f"Running {len(self.benchmark_queries)} benchmark queries...")
        
        start_time = time.time()
        query_results = {}
        
        for i, query in enumerate(self.benchmark_queries):
            logger.info(f"Running query {i+1}/{len(self.benchmark_queries)}: {query}")
            
            query_start_time = time.time()
            
            try:
                result = await agent.query(query)
                
                query_time = time.time() - query_start_time
                self.metrics['query_times'].append(query_time)
                
                query_results[query] = {
                    'success': result.success,
                    'execution_time': query_time,
                    'answer_length': len(result.answer),
                    'frames_analyzed': len(result.frames),
                    'error': result.error
                }
                
                logger.info(f"✓ Query completed in {query_time:.3f}s")
                
            except Exception as e:
                logger.error(f"✗ Query failed: {e}")
                query_results[query] = {
                    'success': False,
                    'execution_time': time.time() - query_start_time,
                    'error': str(e)
                }
        
        self.metrics['total_query_time'] = time.time() - start_time
        
        logger.info(f"All queries completed in {self.metrics['total_query_time']:.2f}s")
        logger.info(f"Average query time: {statistics.mean(self.metrics['query_times']):.3f}s")
        
        return query_results
    
    async def run_batch_benchmark(self, agent: RoboDMAgent) -> Dict[str, Any]:
        """Run batch query benchmark."""
        logger.info("Running batch query benchmark...")
        
        # Select a subset of queries for batch testing
        batch_queries = self.benchmark_queries[:5]  # First 5 queries
        
        start_time = time.time()
        
        try:
            batch_results = await agent.batch_query(batch_queries)
            
            batch_time = time.time() - start_time
            
            successful_queries = sum(1 for r in batch_results if r.success)
            
            batch_metrics = {
                'total_time': batch_time,
                'queries_processed': len(batch_queries),
                'successful_queries': successful_queries,
                'average_time_per_query': batch_time / len(batch_queries),
                'throughput': len(batch_queries) / batch_time
            }
            
            logger.info(f"Batch benchmark completed:")
            logger.info(f"  Total time: {batch_time:.3f}s")
            logger.info(f"  Successful queries: {successful_queries}/{len(batch_queries)}")
            logger.info(f"  Throughput: {batch_metrics['throughput']:.2f} queries/second")
            
            return batch_metrics
            
        except Exception as e:
            logger.error(f"Batch benchmark failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, query_results: Dict[str, Any], batch_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("DROID DATASET BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Ingestion metrics
        report.append("INGESTION METRICS:")
        report.append(f"  Total trajectories: {len(self.metrics['trajectory_sizes'])}")
        report.append(f"  Total ingestion time: {self.metrics['total_ingestion_time']:.2f}s")
        report.append(f"  Average ingestion time per trajectory: {statistics.mean(self.metrics['ingestion_times']):.3f}s")
        report.append(f"  Average trajectory size: {statistics.mean(self.metrics['trajectory_sizes']):.2f}MB")
        report.append(f"  Total data size: {sum(self.metrics['trajectory_sizes']):.2f}MB")
        report.append(f"  Parallel workers: {self.max_workers}")
        report.append("")
        
        # Query metrics
        report.append("QUERY PERFORMANCE METRICS:")
        report.append(f"  Total queries: {len(self.metrics['query_times'])}")
        report.append(f"  Total query time: {self.metrics['total_query_time']:.2f}s")
        report.append(f"  Average query time: {statistics.mean(self.metrics['query_times']):.3f}s")
        report.append(f"  Median query time: {statistics.median(self.metrics['query_times']):.3f}s")
        report.append(f"  Min query time: {min(self.metrics['query_times']):.3f}s")
        report.append(f"  Max query time: {max(self.metrics['query_times']):.3f}s")
        report.append("")
        
        # Batch metrics
        if 'error' not in batch_metrics:
            report.append("BATCH PROCESSING METRICS:")
            report.append(f"  Batch queries: {batch_metrics['queries_processed']}")
            report.append(f"  Batch total time: {batch_metrics['total_time']:.3f}s")
            report.append(f"  Average time per query (batch): {batch_metrics['average_time_per_query']:.3f}s")
            report.append(f"  Throughput: {batch_metrics['throughput']:.2f} queries/second")
            report.append("")
        
        # Query success rates
        successful_queries = sum(1 for r in query_results.values() if r['success'])
        report.append("QUERY SUCCESS RATES:")
        report.append(f"  Successful queries: {successful_queries}/{len(query_results)}")
        report.append(f"  Success rate: {successful_queries/len(query_results)*100:.1f}%")
        report.append("")
        
        # Individual query results
        report.append("INDIVIDUAL QUERY RESULTS:")
        for query, result in query_results.items():
            status = "✓" if result['success'] else "✗"
            report.append(f"  {status} {query[:50]}... ({result['execution_time']:.3f}s)")
        report.append("")
        
        # Performance recommendations
        report.append("PERFORMANCE RECOMMENDATIONS:")
        avg_query_time = statistics.mean(self.metrics['query_times'])
        if avg_query_time > 5.0:
            report.append("  ⚠️  Average query time is high (>5s). Consider:")
            report.append("     - Implementing caching")
            report.append("     - Using batch processing")
            report.append("     - Optimizing frame selection")
        elif avg_query_time > 2.0:
            report.append("  ⚠️  Query time is moderate (>2s). Consider:")
            report.append("     - Frame selection optimization")
            report.append("     - Query result caching")
        else:
            report.append("  ✓ Query performance is good")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark pipeline."""
        logger.info("Starting DROID dataset benchmark...")
        
        # Step 1: Ingest trajectories
        trajectory_paths = await self.ingest_droid_trajectories()
        
        if not trajectory_paths:
            logger.error("No trajectories ingested. Benchmark failed.")
            return {'error': 'No trajectories ingested'}
        
        # Step 2: Set up agent
        agent = await self.setup_agent(trajectory_paths)
        
        if agent is None:
            logger.error("Failed to set up agent. Benchmark failed.")
            return {'error': 'Agent setup failed'}
        
        # Step 3: Run individual queries
        query_results = await self.run_benchmark_queries(agent)
        
        # Step 4: Run batch queries
        batch_metrics = await self.run_batch_benchmark(agent)
        
        # Step 5: Generate report
        report = self.generate_report(query_results, batch_metrics)
        
        # Save report
        report_path = self.output_dir / "benchmark_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save metrics as JSON
        metrics_path = self.output_dir / "benchmark_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'query_results': query_results,
                'batch_metrics': batch_metrics
            }, f, indent=2, default=str)
        
        logger.info(f"Benchmark completed. Report saved to: {report_path}")
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Print report
        print(report)
        
        return {
            'trajectory_paths': trajectory_paths,
            'query_results': query_results,
            'batch_metrics': batch_metrics,
            'report_path': str(report_path),
            'metrics_path': str(metrics_path)
        }


async def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="DROID Dataset Benchmark for RoboDM Agentic")
    parser.add_argument("--num-trajectories", type=int, default=1000, 
                       help="Number of trajectories to ingest (default: 1000)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for trajectories and reports")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel workers for ingestion (default: 4)")
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = DROIDBenchmark(
        num_trajectories=args.num_trajectories,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Run benchmark
    results = await benchmark.run_full_benchmark()
    
    if 'error' in results:
        logger.error(f"Benchmark failed: {results['error']}")
        return 1
    
    logger.info("Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 