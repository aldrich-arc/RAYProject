import numpy as np
import tkinter as tk
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

class MazeWindow:
    def __init__(self, maze):
        self.root = tk.Tk()
        self.root.title('Ray Parallel Maze')
        self.maze = maze
        self.labels = np.zeros(self.maze.shape).tolist()
        self.plotBackground()
    def plotBackground(self):
        for i, row in enumerate(self.maze.tolist()):
            for j, element in enumerate(row):
                bg = 'black' if element == 1 else 'red' if element == 2 else 'green' if element == -1 else 'white'
                self.labels[i][j] = tk.Label(self.root, foreground='blue', background=bg, width=2, height=1, relief='ridge', font='? 20 bold')
                self.labels[i][j].grid(row=i, column=j)
    def target(self, indexes):
        for label in [item for row in self.labels for item in row]:
            label.config(text='')
        self.labels[indexes[0]][indexes[1]].config(text = 'o')
        self.root.update()

class Agent:
    def __init__(self, maze, initState):
        self.state = initState
        self.maze = maze
        self.actionList = ['up', 'down', 'left', 'right']
        self.actionDict = {element : index for index, element in enumerate(self.actionList)}
        Q = np.zeros(self.maze.shape).tolist()
        for i, row in enumerate(Q):
            for j, _ in enumerate(row):
                Q[i][j] = [0, 0, 0, 0]
        self.QTable = np.array(Q, dtype='f')
    def getAction(self, eGreddy=0.8):
        if random.random() > eGreddy: return random.choice(self.actionList)
        else: return self.actionList[self.QTable[self.state].tolist().index(max(self.QTable[self.state].tolist()))]
    def getNextMaxQ(self, state): return max(np.array(self.QTable[state]))
    def updateQTable(self, action, nextState, reward, lr=0.7, gamma=0.9):
        Qs = self.QTable[self.state]
        Qsa = Qs[self.actionDict[action]]
        Qs[self.actionDict[action]] = (1 - lr) * Qsa + lr * (reward + gamma *(self.getNextMaxQ(nextState)))

class Environment:
    def getNextState(self, state, action):
        row, column = state
        if action == 'up': row -= 1
        elif action == 'down': row += 1
        elif action == 'left': column -= 1
        elif action == 'right': column += 1
        nextState = (row, column)
        if row < 0 or column < 0 or row >= maze.shape[0] or column >= maze.shape[1] or maze[row, column] == 1:
            return [state, False]
        elif maze[row, column] == 2: return [nextState, True]
        else: return [nextState, False]
    def doAction(self, state, action):
        nextState, result = self.getNextState(state, action)
        reward = -10 if nextState == state else (100 if result else -1)
        return [reward, nextState, result]


@ray.remote
def train_task(worker_id, maze, num_episodes=25):
    """
    Ray 遠端任務：每個 Worker 執行獨立的訓練。
    只有 worker_id == 0 的時候會啟動 GUI 視窗。
    """
    initState = (np.where(maze==-1)[0][0], np.where(maze==-1)[1][0])
    agent = Agent(maze, initState)
    env = Environment()
    
    m = None
    if worker_id == 0:
        print(f"Worker {worker_id}: 啟動圖形監控視窗...")
        m = MazeWindow(maze)

    best_steps = float('inf')

    for j in range(num_episodes):
        agent.state = initState
        if m: m.target(agent.state)
        
        i = 0
        while True:
            i += 1
            action = agent.getAction(0.9)
            reward, nextState, result = env.doAction(agent.state, action)
            agent.updateQTable(action, nextState, reward)
            agent.state = nextState
            
            if m: 
                m.target(agent.state)
                time.sleep(0.01)

            if result:
                if i < best_steps:
                    best_steps = i
                break
        
        if worker_id == 0:
            print(f"ID 0 訓練進度: {j+1}/{num_episodes} (本局步數: {i})")

    if m: m.root.destroy()
    
    return {"id": worker_id, "steps": best_steps, "qtable": agent.QTable}

if __name__ == "__main__":
    try:
        ray.init(address='auto')
    except:
        ray.init()

    print(" 啟動並行訓練 (20 個執行緒)...")
    start_time = time.time()
    
    num_workers = 20
    futures = [train_task.remote(i, maze, num_episodes=25) for i in range(num_workers)]
    
    all_results = ray.get(futures)
    
    best_worker = min(all_results, key=lambda x: x['steps'])
    
    print(f"\n訓練完成！耗時: {time.time() - start_time:.2f} 秒")
    print(f"最優 Worker ID: {best_worker['id']}，最少步數: {best_worker['steps']}")

    # 4. 儲存結果為 ray_best_qtable
    np.save('ray_best_qtable.npy', best_worker['qtable'])
    print(f"最優模型已儲存至: ray_best_qtable.npy")