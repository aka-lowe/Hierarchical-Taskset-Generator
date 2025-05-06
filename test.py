#!/usr/bin/env python3
"""
Hierarchical Scheduler Taskset Generator

This script generates realistic tasksets for hierarchical scheduling systems with 
multiple cores, components, and tasks using EDF (Earliest Deadline First) and 
RM (Rate Monotonic) scheduling algorithms.

Usage:
    python hierarchical_taskset_generator.py [options]

Options:
    --num-cores=N        Number of cores in the system (default: 4)
    --num-components=N   Number of components across all cores (default: 8)
    --num-tasks=N        Number of tasks across all components (default: 20)
    --utilization=N      Total system utilization in percent (default: 70)
    --output-dir=DIR     Directory to output the generated files (default: Test_Cases/generated)
    --test-case-name=N   Name of the test case (default: hierarchical-test-case)
    --speed-factor-range=MIN,MAX  Range for core speed factors (default: 0.5,1.5)
    --schedulable        Generate a schedulable system (default: True)
    --seed=N             Random seed for reproducibility (default: None)
"""

import os
import random
import numpy as np
import argparse
import math
import sys
import csv
from typing import List, Dict, Tuple, Optional, Any

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate hierarchical tasksets')
    parser.add_argument('--num-cores', type=int, default=4, 
                        help='Number of cores in the system')
    parser.add_argument('--num-components', type=int, default=8, 
                        help='Number of components across all cores')
    parser.add_argument('--num-tasks', type=int, default=20, 
                        help='Number of tasks across all components')
    parser.add_argument('--utilization', type=float, default=70.0, 
                        help='Total system utilization in percent')
    parser.add_argument('--output-dir', type=str, default='Test_Cases/generated', 
                        help='Directory to output the generated files')
    parser.add_argument('--test-case-name', type=str, default='hierarchical-test-case', 
                        help='Name of the test case')
    parser.add_argument('--speed-factor-range', type=str, default='0.5,1.5', 
                        help='Range for core speed factors (MIN,MAX)')
    parser.add_argument('--unschedulable', action='store_true', 
                        help='Generate an unschedulable system (default is schedulable)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Parse speed factor range
    speed_min, speed_max = map(float, args.speed_factor_range.split(','))
    args.speed_factor_range = (speed_min, speed_max)
    
    return args

def randfixedsum(n: int, u: float, nsets: int = 1, minval: float = 0.001, maxval: float = 0.999) -> np.ndarray:
    """
    Generate random sets of values with a fixed sum.
    
    Based on the RandFixedSum algorithm by Paul Emberson et al.
    
    Args:
        n: Number of values in each set
        u: Target sum of each set
        nsets: Number of sets to generate
        minval: Minimum value allowed
        maxval: Maximum value allowed
        
    Returns:
        A numpy array of shape (nsets, n) containing the random values
    """
    if n <= 0 or u < 0 or nsets <= 0 or minval < 0 or maxval <= minval:
        raise ValueError("Invalid parameters for randfixedsum")
        
    # Scale the sum to [0, n]
    s = u
    
    if s > n * maxval or s < n * minval:
        raise ValueError(f"Sum {s} is outside the feasible range [{n * minval}, {n * maxval}]")
    
    # Generate random points in the simplex
    t = np.random.random((nsets, n-1))
    t = np.sort(t, axis=1)
    t = np.column_stack((np.zeros((nsets, 1)), t, np.ones((nsets, 1))))
    diff = np.diff(t, axis=1)
    
    # Scale values to the desired sum and range
    diff = diff * s / np.sum(diff, axis=1, keepdims=True)
    
    # Constrain values to [minval, maxval]
    if minval > 0 or maxval < 1:
        for i in range(nsets):
            valid = False
            attempts = 0
            while not valid and attempts < 1000:
                diff_i = diff[i]
                if np.all(diff_i >= minval) and np.all(diff_i <= maxval):
                    valid = True
                else:
                    # Regenerate this set
                    t_i = np.random.random(n-1)
                    t_i = np.sort(t_i)
                    t_new = np.concatenate(([0], t_i, [1]))
                    diff_i = np.diff(t_new)
                    diff_i = diff_i * s / np.sum(diff_i)
                    diff[i] = diff_i
                attempts += 1
            
            if not valid:
                # If we can't generate a valid set, fallback to an even distribution
                diff[i] = np.ones(n) * s / n
                # Adjust to stay within bounds
                diff[i] = np.clip(diff[i], minval, maxval)
                # Normalize to ensure sum is s
                diff[i] = diff[i] * s / np.sum(diff[i])
    
    return diff

def generate_periods(n: int, min_period: int = 10, max_period: int = 1000, 
                      harmonic_ratio: float = 0.5) -> List[int]:
    """
    Generate task periods, with some being harmonic.
    
    Args:
        n: Number of periods to generate
        min_period: Minimum period value
        max_period: Maximum period value
        harmonic_ratio: Ratio of tasks with harmonic periods
        
    Returns:
        List of periods
    """
    # Base periods (powers of 2 or small primes multiplied by powers of 2)
    base_periods = [
        25, 50, 100, 200, 400, 800,  # Powers of 2 * 25
        30, 60, 120, 240, 480, 960,  # Powers of 2 * 30
        75, 150, 300, 600,           # Powers of 2 * 75
        10, 20, 40, 80, 160, 320     # Powers of 2 * 10
    ]
    
    # Filter base periods to be within range
    valid_base_periods = [p for p in base_periods if min_period <= p <= max_period]
    
    if not valid_base_periods:
        # If no valid base periods, create some
        valid_base_periods = [min_period * (2 ** i) for i in range(10) 
                              if min_period * (2 ** i) <= max_period]
    
    # Decide how many harmonic periods to generate
    n_harmonic = int(n * harmonic_ratio)
    n_random = n - n_harmonic
    
    # Generate harmonic periods
    harmonic_periods = []
    for _ in range(n_harmonic):
        harmonic_periods.append(random.choice(valid_base_periods))
    
    # Generate random periods
    random_periods = []
    for _ in range(n_random):
        period = random.randint(min_period, max_period)
        random_periods.append(period)
    
    periods = harmonic_periods + random_periods
    random.shuffle(periods)
    
    return periods

def calculate_wcet_from_utilization(utilizations: np.ndarray, periods: List[int]) -> List[int]:
    """
    Calculate Worst-Case Execution Times (WCET) from utilizations and periods.
    
    Args:
        utilizations: Array of task utilizations
        periods: List of task periods
        
    Returns:
        List of WCETs
    """
    wcets = []
    for util, period in zip(utilizations, periods):
        # WCET = Utilization * Period
        wcet = round(util * period)
        # Ensure WCET is at least 1
        wcet = max(1, wcet)
        wcets.append(wcet)
    
    return wcets

def generate_cores(num_cores: int, speed_factor_range: Tuple[float, float]) -> List[Dict[str, Any]]:
    """
    Generate core definitions with varying speed factors.
    
    Args:
        num_cores: Number of cores to generate
        speed_factor_range: Range for core speed factors (min, max)
        
    Returns:
        List of core dictionaries
    """
    cores = []
    for i in range(1, num_cores + 1):
        # Generate a random speed factor within the given range
        speed_factor = round(random.uniform(speed_factor_range[0], speed_factor_range[1]), 2)
        
        # Alternate between EDF and RM for core schedulers
        scheduler = "EDF" if i % 2 == 0 else "RM"
        
        core = {
            "core_id": f"Core_{i}",
            "speed_factor": speed_factor,
            "scheduler": scheduler
        }
        cores.append(core)
    
    return cores

def distribute_utilization(total_util: float, num_cores: int, cores: List[Dict[str, Any]]) -> List[float]:
    """
    Distribute the total utilization among cores based on their speed factors.
    
    Args:
        total_util: Total system utilization (0-1)
        num_cores: Number of cores
        cores: List of core dictionaries
        
    Returns:
        List of core utilizations
    """
    # Calculate weight for each core based on speed factor
    speed_factors = np.array([core["speed_factor"] for core in cores])
    weights = speed_factors / np.sum(speed_factors)
    
    # Distribute utilization according to weights
    core_utils = weights * total_util
    
    return core_utils

def generate_components(num_components: int, num_cores: int, 
                        core_utils: List[float], cores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate component definitions and distribute them among cores.
    
    Args:
        num_components: Number of components to generate
        num_cores: Number of cores
        core_utils: List of core utilizations
        cores: List of core dictionaries
        
    Returns:
        List of component dictionaries
    """
    components = []
    
    # Distribute components among cores
    components_per_core = [0] * num_cores
    for i in range(num_components):
        # Assign to the core with the fewest components
        target_core_idx = components_per_core.index(min(components_per_core))
        components_per_core[target_core_idx] += 1
    
    # Generate component utilizations for each core
    core_component_utils = []
    for i in range(num_cores):
        if components_per_core[i] > 0:
            # Create random utilizations for components on this core
            utils = randfixedsum(components_per_core[i], core_utils[i], nsets=1, 
                                 minval=0.05, maxval=min(0.9, core_utils[i]))[0]
            core_component_utils.append(utils)
        else:
            core_component_utils.append([])
    
    # Generate component names
    component_names = [
        "Camera_Sensor", "Image_Processor", "Bitmap_Processor", "Lidar_Sensor",
        "Control_Unit", "GPS_Sensor", "Communication_Unit", "Proximity_Sensor", 
        "Radar_Sensor", "Sonar_Sensor", "Laser_Sensor", "Infrared_Sensor",
        "Ultraviolet_Sensor", "Thermal_Sensor", "Pressure_Sensor", "Humidity_Sensor",
        "Temperature_Sensor", "Light_Sensor", "Sound_Sensor", "Vibration_Sensor",
        "Motion_Sensor", "Acceleration_Sensor", "Gyroscope_Sensor", "Magnetometer_Sensor",
        "Compass_Sensor", "Altimeter_Sensor", "Barometer_Sensor", "Hygrometer_Sensor",
        "Anemometer_Sensor", "Rain_Gauge_Sensor", "Snow_Gauge_Sensor", "Thermometer_Sensor"
    ]
    
    # Ensure we have enough names
    if num_components > len(component_names):
        for i in range(len(component_names), num_components):
            component_names.append(f"Component_{i+1}")
    
    # Shuffle the names
    random.shuffle(component_names)
    
    # Create components
    component_idx = 0
    for core_idx in range(num_cores):
        core = cores[core_idx]
        for i in range(components_per_core[core_idx]):
            # Choose a scheduler (if core is RM, all components must be RM or EDF)
            if core["scheduler"] == "RM":
                scheduler = random.choice(["RM", "EDF"])
            else:
                scheduler = "EDF"  # If core is EDF, all components must be EDF
            
            # Set budget and period
            util = core_component_utils[core_idx][i]
            
            # Generate budget and period such that budget/period approx. equals utilization
            period = random.randint(5, 40)  # Component period
            budget = max(1, round(util * period))  # Component budget
            
            # Set priority (only needed for RM scheduling at core level)
            priority = ""
            if core["scheduler"] == "RM":
                priority = i  # Lower value means higher priority
            
            component = {
                "component_id": component_names[component_idx],
                "scheduler": scheduler,
                "budget": budget,
                "period": period,
                "core_id": core["core_id"],
                "priority": priority,
                "utilization": util  # Store for task generation (not in actual format)
            }
            
            components.append(component)
            component_idx += 1
    
    return components

def distribute_tasks(num_tasks: int, components: List[Dict[str, Any]]) -> List[int]:
    """
    Distribute tasks among components.
    
    Args:
        num_tasks: Total number of tasks
        components: List of component dictionaries
        
    Returns:
        List containing the number of tasks per component
    """
    num_components = len(components)
    
    # Ensure minimum 1 task per component
    tasks_per_component = [1] * num_components
    remaining_tasks = num_tasks - num_components
    
    if remaining_tasks < 0:
        # If fewer tasks than components, some components will have no tasks
        tasks_per_component = [0] * num_components
        for i in range(num_tasks):
            tasks_per_component[i] = 1
    elif remaining_tasks > 0:
        # Distribute remaining tasks based on component utilization
        utils = np.array([comp["utilization"] for comp in components])
        weights = utils / np.sum(utils)
        
        # Initial distribution based on weights
        additional_tasks = np.zeros(num_components, dtype=int)
        for _ in range(remaining_tasks):
            # Pick a component with probability proportional to weight
            comp_idx = np.random.choice(num_components, p=weights)
            additional_tasks[comp_idx] += 1
        
        tasks_per_component = [1 + additional for additional in additional_tasks]
    
    return tasks_per_component

def generate_tasks(components: List[Dict[str, Any]], tasks_per_component: List[int]) -> List[Dict[str, Any]]:
    """
    Generate tasks for each component.
    
    Args:
        components: List of component dictionaries
        tasks_per_component: List containing the number of tasks per component
        
    Returns:
        List of task dictionaries
    """
    tasks = []
    task_id = 0
    
    for comp_idx, component in enumerate(components):
        num_tasks = tasks_per_component[comp_idx]
        
        if num_tasks == 0:
            continue
        
        # Get component utilization
        comp_util = component["utilization"]
        
        # Generate task utilizations (relative to component)
        task_utils = randfixedsum(num_tasks, comp_util, nsets=1, 
                                  minval=0.01, maxval=min(0.9, comp_util))[0]
        
        # Generate periods
        periods = generate_periods(num_tasks, min_period=5, max_period=300)
        
        # Calculate WCETs
        wcets = calculate_wcet_from_utilization(task_utils, periods)
        
        # Create tasks
        for i in range(num_tasks):
            priority = ""
            if component["scheduler"] == "RM":
                # RM priority based on period (shorter period -> higher priority)
                priority = i  # We'll sort by period later
            
            task = {
                "task_name": f"Task_{task_id}",
                "wcet": wcets[i],
                "period": periods[i],
                "component_id": component["component_id"],
                "priority": priority
            }
            
            tasks.append(task)
            task_id += 1
    
    # Sort tasks by period within each RM component for proper RM priority assignment
    for component in components:
        if component["scheduler"] == "RM":
            # Get tasks for this component
            comp_tasks = [task for task in tasks if task["component_id"] == component["component_id"]]
            
            if comp_tasks:
                # Sort by period (ascending)
                comp_tasks.sort(key=lambda x: x["period"])
                
                # Assign priorities (0 = highest priority)
                for i, task in enumerate(comp_tasks):
                    # Find this task in the main list and update its priority
                    for t in tasks:
                        if t["task_name"] == task["task_name"]:
                            t["priority"] = i
                            break
    
    return tasks

def write_csv_files(cores: List[Dict[str, Any]], components: List[Dict[str, Any]], 
                    tasks: List[Dict[str, Any]], output_dir: str, case_name: str) -> None:
    """
    Write the generated data to CSV files.
    
    Args:
        cores: List of core dictionaries
        components: List of component dictionaries
        tasks: List of task dictionaries
        output_dir: Directory to output the files
        case_name: Name of the test case
    """
    # Create the output directory if it doesn't exist
    full_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Write architecture.csv
    with open(os.path.join(full_output_dir, "architecture.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["core_id", "speed_factor", "scheduler"])
        for core in cores:
            writer.writerow([core["core_id"], core["speed_factor"], core["scheduler"]])
    
    # Write budgets.csv
    with open(os.path.join(full_output_dir, "budgets.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["component_id", "scheduler", "budget", "period", "core_id", "priority"])
        for comp in components:
            # Remove utilization - it's not part of the actual format
            comp_data = comp.copy()
            if "utilization" in comp_data:
                del comp_data["utilization"]
            writer.writerow([
                comp_data["component_id"], 
                comp_data["scheduler"], 
                comp_data["budget"], 
                comp_data["period"], 
                comp_data["core_id"], 
                comp_data["priority"]
            ])
    
    # Write tasks.csv
    with open(os.path.join(full_output_dir, "tasks.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "wcet", "period", "component_id", "priority"])
        for task in tasks:
            writer.writerow([
                task["task_name"], 
                task["wcet"], 
                task["period"], 
                task["component_id"], 
                task["priority"]
            ])
    
    print(f"Generated files written to {full_output_dir}")

def adjust_schedulability(components: List[Dict[str, Any]], tasks: List[Dict[str, Any]], 
                          make_schedulable: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Adjust task parameters to make the system schedulable or unschedulable.
    
    Args:
        components: List of component dictionaries
        tasks: List of task dictionaries
        make_schedulable: True to make system schedulable, False for unschedulable
        
    Returns:
        Updated components and tasks
    """
    if make_schedulable:
        # Make system schedulable by ensuring utilization constraints
        for component in components:
            comp_tasks = [t for t in tasks if t["component_id"] == component["component_id"]]
            
            if not comp_tasks:
                continue
            
            scheduler = component["scheduler"]
            
            if scheduler == "EDF":
                # For EDF, ensure sum of utilizations <= 1
                task_utils = [t["wcet"] / t["period"] for t in comp_tasks]
                total_util = sum(task_utils)
                
                if total_util > 0.9:  # Leave some margin
                    # Scale down WCETs
                    scale_factor = 0.9 / total_util
                    for task in comp_tasks:
                        task_idx = tasks.index(task)
                        tasks[task_idx]["wcet"] = max(1, int(task["wcet"] * scale_factor))
            
            elif scheduler == "RM":
                # For RM, use utilization bound: U <= n(2^(1/n) - 1)
                n = len(comp_tasks)
                if n > 0:
                    rm_bound = n * (2 ** (1/n) - 1)
                    task_utils = [t["wcet"] / t["period"] for t in comp_tasks]
                    total_util = sum(task_utils)
                    
                    if total_util > rm_bound * 0.95:  # Leave some margin
                        # Scale down WCETs
                        scale_factor = rm_bound * 0.95 / total_util
                        for task in comp_tasks:
                            task_idx = tasks.index(task)
                            tasks[task_idx]["wcet"] = max(1, int(task["wcet"] * scale_factor))
    
    else:  # Make unschedulable
        # Pick a random component and increase its tasks' WCETs
        if components:
            component = random.choice(components)
            comp_tasks = [t for t in tasks if t["component_id"] == component["component_id"]]
            
            if comp_tasks:
                # Increase WCETs to exceed schedulability bounds
                scale_factor = 1.5  # Increase by 50%
                for task in comp_tasks:
                    task_idx = tasks.index(task)
                    tasks[task_idx]["wcet"] = max(1, int(task["wcet"] * scale_factor))
    
    return components, tasks

def main():
    """Main function to generate hierarchical tasksets."""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Convert utilization from percentage to fraction
    utilization = args.utilization / 100.0
    
    print(f"Generating hierarchical taskset with parameters:")
    print(f"  Number of cores: {args.num_cores}")
    print(f"  Number of components: {args.num_components}")
    print(f"  Number of tasks: {args.num_tasks}")
    print(f"  Utilization: {args.utilization}%")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Test case name: {args.test_case_name}")
    print(f"  Speed factor range: {args.speed_factor_range}")
    print(f"  Schedulable: {not args.unschedulable}")
    
    # Generate cores
    cores = generate_cores(args.num_cores, args.speed_factor_range)
    
    # Distribute utilization among cores
    core_utils = distribute_utilization(utilization, args.num_cores, cores)
    
    # Generate components
    components = generate_components(args.num_components, args.num_cores, core_utils, cores)
    
    # Distribute tasks among components
    tasks_per_component = distribute_tasks(args.num_tasks, components)
    
    # Generate tasks
    tasks = generate_tasks(components, tasks_per_component)
    
    # Adjust schedulability if needed
    components, tasks = adjust_schedulability(components, tasks, not args.unschedulable)
    
    # Write to CSV files
    write_csv_files(cores, components, tasks, args.output_dir, args.test_case_name)
    
    print("Done!")

if __name__ == "__main__":
    main()