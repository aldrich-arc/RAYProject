import numpy as np
import random
import time
import ray

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
def train_task(worker_id, maze, num_episodes=200):
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
        if worker_id == 0 and (j + 1) % 50 == 0:
            print(f"Worker 0 進度: {j+1}/{num_episodes}, 目前最佳步數: {best_steps}")
            
    return {"id": worker_id, "best_steps": best_steps, "qtable": agent.QTable}

if __name__ == "__main__":
    ray.init()
    
    num_workers = 20
    episodes_per_worker = 200
    
    print(f"開始並行訓練...")
    print(f"配置: {num_workers} 執行緒 x {episodes_per_worker} Episodes = 總計 {num_workers * episodes_per_worker} 次模擬")
    
    start_time = time.time()
    
    futures = [train_task.remote(i, maze, episodes_per_worker) for i in range(num_workers)]
    
    results = ray.get(futures)
    
    execution_time = time.time() - start_time
    best_result = min(results, key=lambda x: x['best_steps'])
    
    print("\n" + "="*30)
    print(f"訓練總耗時: {execution_time:.2f} 秒")
    print(f"全域最優 Worker: ID {best_result['id']}")
    print(f"最短路徑步數: {best_result['best_steps']} 步")
    print("="*30)
    
    np.save('ray_best_qtable.npy', best_result['qtable'])
    print(f"最優 Q-Table 已儲存至 ray_best_qtable.npy")