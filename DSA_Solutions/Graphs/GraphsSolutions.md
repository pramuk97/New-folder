# Graphs DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common graph-based DSA interview questions.

---

## 1. Number of Islands
**Problem:** Count the number of islands in a grid.

```csharp
public int NumIslands(char[][] grid)
{
    int m = grid.Length, n = grid[0].Length, count = 0;
    void Dfs(int i, int j)
    {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
        grid[i][j] = '0';
        Dfs(i + 1, j); Dfs(i - 1, j); Dfs(i, j + 1); Dfs(i, j - 1);
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == '1') { count++; Dfs(i, j); }
    return count;
}
```
**Explanation:** DFS to mark all connected land.
**Time Complexity:** O(mn)
**Space Complexity:** O(mn)

**Dry Run Example:**
Input: grid=[[1,1,0,0],[1,0,0,1],[0,0,1,1],[0,1,0,0]]
Start at (0,0): Dfs marks all connected 1's as 0
Count increases for each new island
Output: 3

---

## 2. Clone Graph
**Problem:** Clone an undirected graph.

```csharp
public Node CloneGraphMethod(Node node)
{
    if (node == null) return null;
    var map = new Dictionary<Node, Node>();
    Node Dfs(Node n)
    {
        if (map.ContainsKey(n)) return map[n];
        var copy = new Node(n.val);
        map[n] = copy;
        foreach (var nei in n.neighbors) copy.neighbors.Add(Dfs(nei));
        return copy;
    }
    return Dfs(node);
}
```
**Explanation:** DFS with hashmap to clone nodes.
**Time Complexity:** O(V + E)
**Space Complexity:** O(V)

**Dry Run Example:**
Input: Node 1 with neighbors 2,4
DFS: clone 1, then 2, then 4, then back to 1
Output: Deep copy of graph

---

## 3. Course Schedule
**Problem:** Can all courses be finished given prerequisites?

```csharp
public bool CanFinish(int numCourses, int[][] prerequisites)
{
    var graph = new List<int>[numCourses];
    for (int i = 0; i < numCourses; i++) graph[i] = new List<int>();
    foreach (var pre in prerequisites) graph[pre[0]].Add(pre[1]);
    var visited = new int[numCourses];
    bool Dfs(int v)
    {
        if (visited[v] == 1) return false;
        if (visited[v] == 2) return true;
        visited[v] = 1;
        foreach (var nei in graph[v]) if (!Dfs(nei)) return false;
        visited[v] = 2;
        return true;
    }
    for (int i = 0; i < numCourses; i++) if (!Dfs(i)) return false;
    return true;
}
```
**Explanation:** DFS cycle detection for topological sort.
**Time Complexity:** O(V + E)
**Space Complexity:** O(V + E)

**Dry Run Example:**
Input: numCourses=2, prerequisites=[[1,0]]
Graph: 1->0
DFS: visit 0, then 1, no cycle
Output: true

---

## 4. Pacific Atlantic Water Flow
**Problem:** Find cells that can flow to both Pacific and Atlantic oceans.

```csharp
public IList<IList<int>> PacificAtlantic(int[][] heights)
{
    int m = heights.Length, n = heights[0].Length;
    var pacific = new bool[m, n];
    var atlantic = new bool[m, n];
    var res = new List<IList<int>>();
    void Dfs(int i, int j, bool[,] ocean, int prev)
    {
        if (i < 0 || i >= m || j < 0 || j >= n || ocean[i, j] || heights[i][j] < prev) return;
        ocean[i, j] = true;
        Dfs(i + 1, j, ocean, heights[i][j]);
        Dfs(i - 1, j, ocean, heights[i][j]);
        Dfs(i, j + 1, ocean, heights[i][j]);
        Dfs(i, j - 1, ocean, heights[i][j]);
    }
    for (int i = 0; i < m; i++) { Dfs(i, 0, pacific, int.MinValue); Dfs(i, n - 1, atlantic, int.MinValue); }
    for (int j = 0; j < n; j++) { Dfs(0, j, pacific, int.MinValue); Dfs(m - 1, j, atlantic, int.MinValue); }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (pacific[i, j] && atlantic[i, j]) res.Add(new List<int> { i, j });
    return res;
}
```
**Explanation:** DFS from ocean borders, intersection is answer.
**Time Complexity:** O(mn)
**Space Complexity:** O(mn)

**Dry Run Example:**
Input: heights=[[1,2,2,3],[3,2,3,4],[2,4,5,3],[6,7,1,4]]
DFS from Pacific and Atlantic borders, mark reachable
Intersection: cells that can reach both
Output: [[0,3],[1,3],[3,0],[3,1],[3,2],[2,2],[2,3]]

---

## 5. Rotting Oranges
**Problem:** Minimum time to rot all oranges.

```csharp
public int OrangesRotting(int[][] grid)
{
    int m = grid.Length, n = grid[0].Length, fresh = 0, minutes = 0;
    var queue = new Queue<(int, int)>();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == 2) queue.Enqueue((i, j));
            else if (grid[i][j] == 1) fresh++;
    int[][] dirs = { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 0, -1 }, new[] { -1, 0 } };
    while (queue.Count > 0 && fresh > 0)
    {
        int size = queue.Count;
        for (int i = 0; i < size; i++)
        {
            var (x, y) = queue.Dequeue();
            foreach (var d in dirs)
            {
                int nx = x + d[0], ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1)
                {
                    grid[nx][ny] = 2;
                    queue.Enqueue((nx, ny));
                    fresh--;
                }
            }
        }
        minutes++;
    }
    return fresh == 0 ? minutes : -1;
}
```
**Explanation:** BFS to spread rot, count minutes.
**Time Complexity:** O(mn)
**Space Complexity:** O(mn)

**Dry Run Example:**
Input: grid=[[2,1,1],[1,1,0],[0,1,1]]
Queue: all rotten oranges
Minute 1: rot adjacent fresh oranges
Repeat until all are rotten or impossible
Output: 4

---

## 6. Word Ladder
**Problem:** Shortest transformation sequence from beginWord to endWord.

```csharp
public int LadderLength(string beginWord, string endWord, IList<string> wordList)
{
    var set = new HashSet<string>(wordList);
    if (!set.Contains(endWord)) return 0;
    var queue = new Queue<(string, int)>();
    queue.Enqueue((beginWord, 1));
    while (queue.Count > 0)
    {
        var (word, len) = queue.Dequeue();
        if (word == endWord) return len;
        for (int i = 0; i < word.Length; i++)
        {
            var arr = word.ToCharArray();
            for (char c = 'a'; c <= 'z'; c++)
            {
                arr[i] = c;
                var next = new string(arr);
                if (set.Contains(next))
                {
                    queue.Enqueue((next, len + 1));
                    set.Remove(next);
                }
            }
        }
    }
    return 0;
}
```
**Explanation:** BFS, try all one-letter transformations.
**Time Complexity:** O(N * 26^L)
**Space Complexity:** O(N)

**Dry Run Example:**
Input: beginWord="hit", endWord="cog", wordList=["hot","dot","dog","lot","log","cog"]
Queue: ("hit",1)
Transform: hit->hot->dot->dog->cog
Output: 5

---

## 7. Graph Valid Tree
**Problem:** Check if a graph is a valid tree.

```csharp
public bool ValidTree(int n, int[][] edges)
{
    if (edges.Length != n - 1) return false;
    var parent = new int[n];
    for (int i = 0; i < n; i++) parent[i] = i;
    int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
    foreach (var e in edges)
    {
        int x = Find(e[0]), y = Find(e[1]);
        if (x == y) return false;
        parent[x] = y;
    }
    return true;
}
```
**Explanation:** Union-find to check for cycles and connectivity.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: n=5, edges=[[0,1],[0,2],[0,3],[1,4]]
Union-find: connect all nodes, no cycles
Output: true

---

## 8. Network Delay Time
**Problem:** Time for all nodes to receive signal from a source.

```csharp
public int NetworkDelayTimeMethod(int[][] times, int n, int k)
{
    var graph = new List<(int, int)>[n + 1];
    for (int i = 1; i <= n; i++) graph[i] = new List<(int, int)>();
    foreach (var t in times) graph[t[0]].Add((t[1], t[2]));
    var dist = new int[n + 1];
    Array.Fill(dist, int.MaxValue);
    dist[k] = 0;
    var pq = new SortedSet<(int, int)>();
    pq.Add((0, k));
    while (pq.Count > 0)
    {
        var (d, u) = pq.Min; pq.Remove(pq.Min);
        foreach (var (v, w) in graph[u])
        {
            if (dist[v] > d + w)
            {
                pq.Remove((dist[v], v));
                dist[v] = d + w;
                pq.Add((dist[v], v));
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; i++)
        if (dist[i] == int.MaxValue) return -1;
        else ans = Math.Max(ans, dist[i]);
    return ans;
}
```
**Explanation:** Dijkstra's algorithm for shortest path.
**Time Complexity:** O(E log V)
**Space Complexity:** O(V + E)

**Dry Run Example:**
Input: times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2
Build graph, Dijkstra from 2
dist=[inf,1,0,1,2]
Output: 2

---

## 9. Find Eventual Safe States
**Problem:** Find all eventual safe nodes in a graph.

```csharp
public IList<int> EventualSafeNodes(int[][] graph)
{
    int n = graph.Length;
    var color = new int[n];
    var res = new List<int>();
    bool Dfs(int node)
    {
        if (color[node] > 0) return color[node] == 2;
        color[node] = 1;
        foreach (var nei in graph[node])
            if (!Dfs(nei)) return false;
        color[node] = 2;
        return true;
    }
    for (int i = 0; i < n; i++) if (Dfs(i)) res.Add(i);
    return res;
}
```
**Explanation:** DFS with coloring to find safe nodes.
**Time Complexity:** O(V + E)
**Space Complexity:** O(V)

**Dry Run Example:**
Input: graph=[[1,2],[2,3],[5],[0],[5],[],[]]
DFS with coloring, find safe nodes
Output: [2,4,5,6]

---

## 10. Number of Connected Components
**Problem:** Count the number of connected components in a graph.

```csharp
public int CountComponents(int n, int[][] edges)
{
    var parent = new int[n];
    for (int i = 0; i < n; i++) parent[i] = i;
    int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
    foreach (var e in edges)
    {
        int x = Find(e[0]), y = Find(e[1]);
        if (x != y) parent[x] = y;
    }
    var set = new HashSet<int>();
    for (int i = 0; i < n; i++) set.Add(Find(i));
    return set.Count;
}
```
**Explanation:** Union-find to count connected components.
**Time Complexity:** O(n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: n=5, edges=[[0,1],[1,2],[3,4]]
Union-find: connect 0-1-2, 3-4
Count unique parents
Output: 2

---
