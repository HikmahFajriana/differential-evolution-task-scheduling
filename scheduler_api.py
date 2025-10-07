from flask import Flask, request, jsonify
import numpy as np
import time
import random
from datetime import datetime

app = Flask(__name__)

# ============================================
# DIFFERENTIAL EVOLUTION ALGORITHM
# ============================================

class DifferentialEvolution:
    def __init__(self, tasks, nodes, pop_size=50, max_iter=100):
        self.tasks = tasks
        self.nodes = nodes
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.num_tasks = len(tasks)
        self.num_nodes = len(nodes)
        
    def fitness(self, solution):
        """Calculate makespan (fitness function)"""
        node_times = {node['id']: 0 for node in self.nodes}
        
        for task_idx, node_idx in enumerate(solution):
            task = self.tasks[task_idx]
            node = self.nodes[int(node_idx)]
            
            # Calculate execution time based on task complexity and node capacity
            exec_time = task['complexity'] * (10 / node['capacity'])
            node_times[node['id']] += exec_time
        
        # Makespan is the maximum completion time
        makespan = max(node_times.values())
        return makespan
    
    def optimize(self):
        """Run Differential Evolution optimization"""
        # Initialize population
        population = np.random.randint(0, self.num_nodes, 
                                      size=(self.pop_size, self.num_tasks))
        
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover probability
        
        best_solution = None
        best_fitness = float('inf')
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # Mutation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = random.sample(indices, 3)
                
                mutant = np.clip(
                    population[a] + F * (population[b] - population[c]),
                    0, self.num_nodes - 1
                ).astype(int)
                
                # Crossover
                trial = np.copy(population[i])
                for j in range(self.num_tasks):
                    if random.random() < CR:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = self.fitness(trial)
                current_fitness = self.fitness(population[i])
                
                if trial_fitness < current_fitness:
                    population[i] = trial
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial.copy()
        
        return best_solution, best_fitness

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'DE Task Scheduler API',
        'endpoints': ['/schedule', '/health']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/schedule', methods=['POST'])
def schedule_tasks():
    """Main scheduling endpoint"""
    try:
        data = request.json
        tasks = data.get('tasks', [])
        nodes = data.get('nodes', [])
        
        if not tasks or not nodes:
            return jsonify({'error': 'Tasks and nodes are required'}), 400
        
        # Run DE optimization
        start_time = time.time()
        de = DifferentialEvolution(tasks, nodes)
        solution, makespan = de.optimize()
        optimization_time = time.time() - start_time
        
        # Create schedule
        schedule = {}
        for node in nodes:
            schedule[node['id']] = []
        
        for task_idx, node_idx in enumerate(solution):
            node_id = nodes[int(node_idx)]['id']
            schedule[node_id].append(tasks[task_idx])
        
        return jsonify({
            'success': True,
            'schedule': schedule,
            'makespan': float(makespan),
            'optimization_time': optimization_time,
            'algorithm': 'Differential Evolution',
            'total_tasks': len(tasks),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/schedule/fcfs', methods=['POST'])
def schedule_fcfs():
    """FCFS baseline for comparison"""
    try:
        data = request.json
        tasks = data.get('tasks', [])
        nodes = data.get('nodes', [])
        
        if not tasks or not nodes:
            return jsonify({'error': 'Tasks and nodes are required'}), 400
        
        start_time = time.time()
        
        # Simple FCFS: assign tasks round-robin
        schedule = {node['id']: [] for node in nodes}
        node_times = {node['id']: 0 for node in nodes}
        
        for i, task in enumerate(tasks):
            node = nodes[i % len(nodes)]
            schedule[node['id']].append(task)
            exec_time = task['complexity'] * (10 / node['capacity'])
            node_times[node['id']] += exec_time
        
        makespan = max(node_times.values())
        optimization_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'schedule': schedule,
            'makespan': float(makespan),
            'optimization_time': optimization_time,
            'algorithm': 'FCFS',
            'total_tasks': len(tasks),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)