import sys
import numpy as np
import random
import time
import webbrowser
import subprocess
import ray
import psutil
import matplotlib.pyplot as plt
import threading

maze = np.array([
    [-1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [ 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [ 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [ 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
])

class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu_history = []
        self.timestamps = []

    def run(self):
        start_time = time.time()
        # 先呼叫一次以初始化
        psutil.cpu_percent(percpu=True)
        
        while not self.stop_flag.is_set():
            current_cpu = psutil.cpu_percent(percpu=True)
            self.cpu_history.append(current_cpu)
            self.timestamps.append(time.time() - start_time)
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()

    def generate_plot(self, filename="cpu_profile.png"):
        data = np.array(self.cpu_history) # 形狀: (時間點, 核心數)
        plt.figure(figsize=(12, 6))
        
        num_cores = data.shape[1]
        for i in range(num_cores):
            plt.plot(self.timestamps, data[:, i], alpha=0.6, label=f"Core {i}" if num_cores <= 8 else "")
        
        plt.title(f"CPU Utilization Profile ({num_cores} Cores)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Utilization (%)")
        plt.ylim(0, 105)
        if num_cores <= 8: plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"資源統計圖已儲存至: {filename}")

def init_ray(num_cpus):
    if ray.is_initialized():
        print("Ray 已經啟動。")
        return

    try:
        ray.init(address='auto', ignore_reinit_error=True)
        print("成功連接到【後台常駐】的 Ray")
    except (ConnectionError, Exception):
        print(f"找不到後台 Ray，正在啟動Ray (CPUs: {num_cpus})...")
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=True, 
            dashboard_host="127.0.0.1", 
            dashboard_port=8265,
            ignore_reinit_error=True
        )
    
    url = "http://127.0.0.1:8265"
    print(f"Ray Dashboard 網址: {url}")
    
    try:
        subprocess.run(['powershell.exe', '/c', f'start {url}'], check=True)
    except Exception:
        webbrowser.open(url)

class Agent:
    def __init__(self, maze, initState):
        self.state = initState
        self.maze = maze
        self.actionList = ['up', 'down', 'left', 'right']
        self.actionDict = {element : index for index, element in enumerate(self.actionList)}
        self.QTable = np.zeros((*self.maze.shape, 4), dtype='f')
        
    def getAction(self, eGreddy=0.8):
        if random.random() > eGreddy:
            return random.choice(self.actionList)
        else:
            Qsa = self.QTable[self.state].tolist()
            return self.actionList[Qsa.index(max(Qsa))]
            
    def getNextMaxQ(self, state):
        return max(self.QTable[state])
        
    def updateQTable(self, action, nextState, reward, lr=0.1, gamma=0.9):
        Qs = self.QTable[self.state]
        action_idx = self.actionDict[action]
        Qsa = Qs[action_idx]
        Qs[action_idx] = (1 - lr) * Qsa + lr * (reward + gamma * self.getNextMaxQ(nextState))

class Environment:
    def doAction(self, state, action):
        row, column = state
        if action == 'up': row -= 1
        elif action == 'down': row += 1
        elif action == 'left': column -= 1
        elif action == 'right': column += 1
        
        nextState = (row, column)
        if row < 0 or column < 0 or row >= maze.shape[0] or column >= maze.shape[1] or maze[row, column] == 1:
            return -10, state, False
        elif maze[row, column] == 2:
            return 100, nextState, True
        else:
            return -1, nextState, False

@ray.remote
def train_task(worker_id, maze, num_episodes):
    initState = (np.where(maze==-1)[0][0], np.where(maze==-1)[1][0])
    agent = Agent(maze, initState)
    env = Environment()
    
    best_steps = float('inf')
    
    for j in range(num_episodes):
        agent.state = initState
        steps = 0
        while True:
            steps += 1
            action = agent.getAction(0.9)
            reward, nextState, result = env.doAction(agent.state, action)
            agent.updateQTable(action, nextState, reward)
            agent.state = nextState
            if result:
                if steps < best_steps:
                    best_steps = steps
                break
        
        # 每 50 次 Episode 印一次進度 (僅限 Worker 0)
        #if worker_id == 0 and (j + 1) % 50 == 0:
        #    print(f"Worker 0 進度: {j+1}/{num_episodes}, 目前最佳步數: {best_steps}")
            
    return {"id": worker_id, "best_steps": best_steps, "qtable": agent.QTable}


if __name__ == "__main__":
    total_episodes = 600000
    num_workers = 10

    if len(sys.argv) > 1:
        try:
            input_num = int(sys.argv[1])
            if input_num > 0:
                num_workers = input_num
            else:
                print(f"輸入值 {input_num} 小於等於 0，將使用預設值 10")
        except ValueError:
            print(f"格式錯誤，將使用預設值 10")
    if len(sys.argv) > 2:
        total_episodes = int(sys.argv[2])

    episodes_per_worker = total_episodes // num_workers
    remainder = total_episodes % num_workers

    init_ray(num_workers)
    
    print(f"   - 總模擬目標：{total_episodes} 次")
    print(f"   - 使用執行緒：{num_workers} 個")

    monitor = ResourceMonitor(interval=0.2)
    monitor.start()
    
    start_time = time.time()
    
    futures = []
    for i in range(num_workers):
        count = episodes_per_worker + (remainder if i == num_workers - 1 else 0)
        futures.append(train_task.remote(i, maze, count))
    
    results = ray.get(futures)

    monitor.stop()
    monitor.join()
    
    execution_time = time.time() - start_time
    best_result = min(results, key=lambda x: x['best_steps'])
    
    print("\n" + "="*30)
    print(f"訓練總耗時: {execution_time:.2f} 秒")
    print(f"全域最優 Worker: ID {best_result['id']}")
    print(f"最短路徑步數: {best_result['best_steps']} 步")
    print("="*30)
    
    np.save('ray_best_qtable.npy', best_result['qtable'])
    print(f"最優 Q-Table 已儲存至 ray_best_qtable.npy")

    plot_file = "resource_usage.png"
    monitor.generate_plot(plot_file)
    
    try:
        subprocess.run(['powershell.exe', '/c', f'start {plot_file}'], check=True)
    except:
        print("無法自動開啟圖片，請手動查看 resource_usage.png")

    ray.shutdown()